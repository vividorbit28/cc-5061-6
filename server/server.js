// Minimal zero-dependency HTTP server:
//   - static files from /web and generated artifacts from /out
//   - JSON API under /api
//   - job-based inversion with polled progress (the CG solve is CPU-bound)

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';

import { GEOM, makeClip, MOTION_PRESETS, DEFAULTS } from '../src/synth.js';
import { invert, residualStats, compareHypotheses, analyzeClip } from '../src/pipeline.js';
import { evaluate, autoLambda, estimateNoise } from '../src/evaluate.js';
import { encodePNG, encodePNGColor, encodeMP4 } from './media.js';
import { turbo } from '../src/util.js';
import { kernelsForUI } from './render.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const WEB = path.join(ROOT, 'web');
const OUT = path.join(ROOT, 'out');
const PORT = Number(process.env.PORT || 5173);

fs.mkdirSync(OUT, { recursive: true });

// ---------------------------------------------------------------------------
// Clip catalog. Clips are deterministic functions of their spec; the truth is
// stored in memory here and is NEVER sent to the browser or to invert().
// ---------------------------------------------------------------------------

const CLIPS = new Map();

function registerClip(id, spec, title) {
  CLIPS.set(id, { id, spec, title });
}

const CATALOG = [
  { id: 'pan-clean', title: 'Pan · clean · short exposure', opts: { seed: 3, motionType: 'pan', exposureKind: 'uniform', rho: 0.75, noiseSigma: 0.0, quantize: false } },
  { id: 'pan-long', title: 'Pan · long exposure (rho 0.95)', opts: { seed: 3, motionType: 'pan', exposureKind: 'centered', rho: 0.95, noiseSigma: 0.003, quantize: true } },
  { id: 'diag', title: 'Diagonal pan · ramp shutter', opts: { seed: 7, motionType: 'diag', exposureKind: 'ramp', rho: 0.8, noiseSigma: 0.003, quantize: true } },
  { id: 'fast', title: 'Fast pan · strong blur', opts: { seed: 11, motionType: 'fast', exposureKind: 'uniform', rho: 0.9, noiseSigma: 0.003, quantize: true } },
  { id: 'shake', title: 'Pan + shake · non-constant velocity', opts: { seed: 5, motionType: 'shake', exposureKind: 'uniform', rho: 0.8, noiseSigma: 0.003, quantize: true } },
  { id: 'arc', title: 'Shutter arc · rotation about distant pivot', opts: { seed: 9, motionType: 'arc', exposureKind: 'centered', rho: 0.8, noiseSigma: 0.003, quantize: true } },
  { id: 'stationary', title: 'Stationary · long exposure', opts: { seed: 13, motionType: 'stationary', exposureKind: 'centered', rho: 0.6, noiseSigma: 0.003, quantize: true } },
  { id: 'noisy', title: 'Pan · high noise (sigma 0.012)', opts: { seed: 17, motionType: 'pan', exposureKind: 'uniform', rho: 0.85, noiseSigma: 0.012, quantize: true } },
];

for (const c of CATALOG) {
  registerClip(c.id, { ...DEFAULTS, frames: 7, refIndex: 3, ...c.opts }, c.title);
}

const clipCache = new Map();
function getClip(id) {
  if (!CLIPS.has(id)) throw new Error(`unknown clip ${id}`);
  if (!clipCache.has(id)) {
    const entry = CLIPS.get(id);
    clipCache.set(id, makeClip(entry.spec));
  }
  return clipCache.get(id);
}

function truthOf(id) {
  const clip = getClip(id);
  return { latent: clip.latent, scene: clip.scene };
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

function grayPNG(img, w, h) {
  return 'data:image/png;base64,' + encodePNG(img, w, h).toString('base64');
}

function colorPNG(rgb, w, h) {
  return 'data:image/png;base64,' + encodePNGColor(rgb, w, h).toString('base64');
}

function arrayToDisplay(img) {
  return Array.from(img, (v) => Math.round(Math.max(0, Math.min(1, v)) * 1000) / 1000);
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

const jobs = new Map();
const JOB_TTL_MS = 30 * 60 * 1000;

function newJob() {
  const id = crypto.randomBytes(8).toString('hex');
  const job = {
    id, state: 'queued', progress: 0, stage: 'queued', error: null,
    createdAt: Date.now(), result: null,
  };
  jobs.set(id, job);
  return job;
}

// Housekeeping: drop old jobs.
setInterval(() => {
  const now = Date.now();
  for (const [id, j] of jobs) if (now - j.createdAt > JOB_TTL_MS) jobs.delete(id);
}, 60 * 1000).unref();

async function runInversionJob(job, req) {
  try {
    job.state = 'running';
    const clip = getClip(req.clipId);
    const obs = clip.obs;
    const obsW = GEOM.obsW, obsH = GEOM.obsH;

    job.stage = 'estimating noise'; job.progress = 0.05;
    const noiseEst = estimateNoise(obs[0], obsW, obsH);
    const autoLam = autoLambda(noiseEst);

    const assumption = {
      id: 'user',
      label: 'user assumption',
      kind: req.kind ?? 'linear',
      vx: req.vx ?? 0, vy: req.vy ?? 0,
      amp: req.amp ?? 0, freq: req.freq ?? 0,
      omega: req.omega ?? 0, pivotX: req.pivotX ?? 0, pivotY: req.pivotY ?? 0,
      rho: req.rho ?? 0.8,
      exposureKind: req.exposureKind ?? 'uniform',
      refIndex: req.refIndex ?? clip.refIndex,
      lambda: req.lambda ?? autoLam,
      lambdaMode: req.lambdaMode ?? 'auto',
      tv: req.tv ?? 0,
      maxIter: req.maxIter ?? 50,
    };
    if (assumption.lambdaMode === 'auto') assumption.lambda = autoLam;

    job.stage = 'building exposure operators'; job.progress = 0.15;
    await tick();

    job.stage = 'conjugate-gradient solve'; job.progress = 0.3;
    const main = invert(obs, assumption, {});
    job.progress = 0.7;

    // Sequential hypotheses comparison requested by the UI.
    let comparison = null;
    if (req.compare) {
      job.stage = 'comparing motion hypotheses'; job.progress = 0.75;
      await tick();
      const angle = Math.atan2(assumption.vy, assumption.vx);
      const speed = Math.hypot(assumption.vx, assumption.vy);
      const hyps = [
        { ...assumption, id: 'assumed', label: 'assumed (forward)' },
        { ...assumption, id: 'reversed', label: 'reversed direction', vx: -assumption.vx, vy: -assumption.vy },
        { ...assumption, id: 'stationary', label: 'stationary', vx: 0, vy: 0 },
      ];
      if (speed > 0) {
        hyps.push({
          ...assumption, id: 'slower', label: '0.6x speed',
          vx: 0.6 * speed * Math.cos(angle), vy: 0.6 * speed * Math.sin(angle),
        });
      }
      comparison = compareHypotheses(obs, hyps, {});
      comparison.results = comparison.results.map((r) => ({
        id: r.id, label: r.label, forwardRmse: r.forwardRmse,
        estimateImage: grayPNG(r.estimate, obsW, obsH),
      }));
      comparison.scored = comparison.scored.map((s) => ({ id: s.id, label: s.label, forwardRmse: s.forwardRmse }));
      // Keep a compact per-hypothesis forward residual history for the note.
      comparison.lambdaUsed = assumption.lambda;
    }

    job.stage = 'evaluating forward model'; job.progress = 0.85;
    const evidence = evidenceFor(clip, main);
    job.progress = 0.95;

    job.result = {
      clipId: req.clipId,
      assumption,
      noiseEstimate: noiseEst,
      autoLambda: autoLam,
      estimate: arrayToDisplay(main.estimate),
      support: arrayToDisplay(main.support ?? new Float64Array(obsW * obsH)),
      reblur: main.reblur.map((r) => ({ predicted: arrayToDisplay(r.predicted), residual: arrayToDisplay(r.residual) })),
      residualStats: residualStats(main.reblur, obsW, obsH),
      history: main.history,
      iterations: main.iterations,
      finalRelResidual: main.finalRelResidual,
      kernel: kernelsForUI(main.kernel),
      evidence,
      comparison,
      widths: { w: obsW, h: obsH },
    };
    job.state = 'done';
    job.progress = 1;
    job.stage = 'done';
  } catch (err) {
    job.state = 'failed';
    job.error = err && err.stack ? err.stack.split('\n').slice(0, 6).join('\n') : String(err);
    console.error('job failed', err);
  }
}

function tick() {
  return new Promise((r) => setImmediate(r));
}

// Evidence block: computed with truth available (server-side only), clearly
// labelled in the UI as "synthetic ground truth, evaluation only".
function evidenceFor(clip, result) {
  const ev = evaluate(result.estimate, clip.refSharp, result.reblur, clip.obs, {});
  return {
    psnr: ev.psnr,
    ssim: ev.ssim,
    blurPsnr: ev.blurPsnr,
    gainDb: ev.gainDb,
    forwardRmse: ev.forwardRmse,
    forwardToNoise: ev.forwardToNoise,
    noiseSigmaEstimate: ev.noiseSigmaEstimate,
    ringing: ev.ringing,
    regions: ev.regions,
    note: 'Ground truth is synthetic and used only for scoring; the solver above never receives it.',
  };
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.mp4': 'video/mp4',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
};

function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(body);
}

function sendError(res, code, msg) {
  sendJSON(res, code, { error: msg });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > 2 * 1024 * 1024) { reject(new Error('body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}); }
      catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

async function handleApi(req, res, url) {
  const p = url.pathname;

  if (p === '/api/catalog' && req.method === 'GET') {
    const clips = [...CLIPS.values()].map((c) => ({
      id: c.id, title: c.title,
      spec: {
        motionType: c.spec.motionType, exposureKind: c.spec.exposureKind,
        rho: c.spec.rho, noiseSigma: c.spec.noiseSigma, frames: c.spec.frames,
        motionParams: c.spec.motionParams,
      },
    }));
    return sendJSON(res, 200, {
      clips,
      geometry: GEOM,
      thresholds: { ambiguityForwardRatio: 0.05 },
      defaults: { rho: 0.8, exposureKind: 'uniform', lambdaMode: 'auto', maxIter: 50 },
      presets: Object.fromEntries(Object.entries(MOTION_PRESETS).map(([k, v]) => [k, { label: v.label }])),
    });
  }

  if (p === '/api/clip' && req.method === 'GET') {
    const id = url.searchParams.get('id');
    if (!CLIPS.has(id)) return sendError(res, 404, 'unknown clip');
    const clip = getClip(id);
    const obsN = clip.spec.noiseSigma > 0 ? clip.spec.noiseSigma : 0;
    const frames = clip.obs.map((g, k) => ({
      index: k,
      obs: grayPNG(g, GEOM.obsW, GEOM.obsH),
      // The "sharp frame" for playback only. It is ground truth and is sent
      // ONLY to support the player's clean-sequence track; never to the solver.
      clean: grayPNG(clip.clean[k], GEOM.obsW, GEOM.obsH),
    }));
    // Blur input analysis: blind registration and the UI's candidate assumption.
    const analysis = analyzeClip(clip.obs, { refIndex: clip.refIndex, rho: 0.8 });
    const perFrame = analysis.reg.frames.map((f) => ({ dx: f.dx, dy: f.dy, spread: f.spread, peak: f.peak }));
    return sendJSON(res, 200, {
      id, title: CLIPS.get(id).title,
      frames,
      refIndex: clip.refIndex,
      spec: {
        ...clip.spec,
        motionParams: clip.spec.motionParams,
      },
      registration: {
        refIndex: analysis.reg.refIndex,
        perFrame,
        candidate: analysis.candidate,
        note: 'Blind phase correlation. Blur biases the estimate toward zero; compare with the forward residual of an explicit assumption.',
      },
      truthMeta: {
        // deliberately only metadata, no pixels
        available: true,
        note: 'Synthetic ground truth exists for scoring. It is not used by the solver.',
      },
    });
  }

  if (p === '/api/invert' && req.method === 'POST') {
    let body;
    try { body = await readBody(req); } catch (e) { return sendError(res, 400, 'bad json'); }
    if (!CLIPS.has(body.clipId)) return sendError(res, 404, 'unknown clip');
    const job = newJob();
    runInversionJob(job, body);
    return sendJSON(res, 202, { jobId: job.id });
  }

  if (p === '/api/job' && req.method === 'GET') {
    const id = url.searchParams.get('id');
    const job = jobs.get(id);
    if (!job) return sendError(res, 404, 'unknown job');
    if (job.state === 'done') {
      const width = GEOM.obsW;
      const height = GEOM.obsH;
      const clip = getClip(job.result.clipId);
      const refIdx = job.result.assumption.refIndex ?? clip.refIndex;
      return sendJSON(res, 200, {
        id, state: job.state, progress: job.progress, stage: job.stage, error: job.error,
        result: {
          ...job.result,
          images: {
            estimate: grayPNG(Float64Array.from(job.result.estimate), width, height),
            supportMap: supportPNG(Float64Array.from(job.result.support)),
            refObs: grayPNG(clip.obs[refIdx], width, height),
            residualCombined: colorPNG(combinedResidualRGB(job.result.reblur, job.result.noiseEstimate).rgb, width, height),
            predictedRef: grayPNG(Float64Array.from(job.result.reblur[refIdx].predicted), width, height),
          },
        },
      });
    }
    return sendJSON(res, 200, {
      id, state: job.state, progress: job.progress, stage: job.stage, error: job.error,
    });
  }

  if (p === '/api/truth' && req.method === 'GET') {
    // Evaluation-only frames, explicitly separated from inversion. The UI shows
    // these in the "ground-truth evaluation" panel with a clear warning.
    const id = url.searchParams.get('id');
    if (!CLIPS.has(id)) return sendError(res, 404, 'unknown clip');
    const clip = getClip(id);
    return sendJSON(res, 200, {
      id,
      latent: grayPNG(clip.latent, GEOM.obsW, GEOM.obsH),
      scene: 'data:image/png;base64,' + encodePNG(clip.scene.data, clip.scene.w, clip.scene.h).toString('base64'),
      note: 'Synthetic ground truth. Never read by the solver.',
    });
  }

  if (p === '/api/build' && req.method === 'POST') {
    if (buildJob && buildJob.state === 'running') {
      return sendJSON(res, 409, { error: 'build already running', jobId: buildJob.id });
    }
    const job = { id: crypto.randomBytes(6).toString('hex'), state: 'running', stage: 'starting', detail: '', error: null, started: Date.now() };
    buildJob = job;
    (async () => {
      try {
        const { buildArtifacts } = await import('./artifacts.js');
        job.stage = 'rendering clips';
        const report = await buildArtifacts({
          onProgress: (stage, detail) => { job.stage = stage; job.detail = detail || ''; },
        });
        job.state = 'done';
        job.stage = 'done';
        job.detail = `${report.length} cases → out/`;
      } catch (e) {
        job.state = 'failed';
        job.error = String(e && e.stack || e).split('\n').slice(0, 4).join('\n');
        console.error(e);
      }
    })();
    return sendJSON(res, 202, { jobId: job.id });
  }

  if (p === '/api/build/status' && req.method === 'GET') {
    const id = url.searchParams.get('id');
    if (!buildJob || buildJob.id !== id) return sendError(res, 404, 'unknown build');
    return sendJSON(res, 200, {
      id: buildJob.id, state: buildJob.state, stage: buildJob.stage,
      detail: buildJob.detail, error: buildJob.error,
      elapsedMs: Date.now() - buildJob.started,
    });
  }

  return sendError(res, 404, 'not found');
}

let buildJob = null;

// Support map, contrast-stretched between its 5th and 95th percentile so the
// measured/inferred distinction is visible (values sit in a narrow high band).
function supportPNG(support) {
  const w = GEOM.obsW, h = GEOM.obsH;
  const sorted = Array.from(support).sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
  const hi = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
  const span = Math.max(hi - lo, 1e-3);
  const stretched = new Float64Array(support.length);
  for (let i = 0; i < support.length; i++) {
    stretched[i] = Math.max(0, Math.min(1, (support[i] - lo) / span));
  }
  return grayPNG(stretched, w, h);
}

function combinedResidual(reblur, w, h) {
  const acc = new Float64Array(w * h);
  for (const r of reblur) for (let i = 0; i < w * h; i++) acc[i] += Math.abs(r.residual[i]);
  for (let i = 0; i < w * h; i++) acc[i] /= reblur.length;
  return acc;
}

// Mean |residual| across frames, colourised for display. The scale is tied to
// the estimated sensor noise: the full colour range spans 4 sigma, so a
// well-fit residual (mean |r| ~ 0.8 sigma) shows as deep blue and structured
// misfit stands out. Returns the scale used so the UI can label it.
function combinedResidualRGB(reblur, sigma) {
  const w = GEOM.obsW, h = GEOM.obsH;
  const acc = combinedResidual(reblur, w, h);
  const full = Math.max(4 * (sigma || 0.005), 1e-3);
  const out = new Uint8Array(w * h * 3);
  for (let i = 0; i < acc.length; i++) {
    const t = Math.min(1, acc[i] / full);
    const [r, g, b] = turbo(t);
    out[3 * i] = Math.round(r * 255);
    out[3 * i + 1] = Math.round(g * 255);
    out[3 * i + 2] = Math.round(b * 255);
  }
  return { rgb: out, full };
}

function serveStatic(req, res, url) {
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/') rel = '/index.html';
  const roots = [
    { prefix: '/', dir: WEB },
    { prefix: '/out/', dir: OUT },
  ];
  let file = null;
  if (rel.startsWith('/out/')) {
    file = path.join(OUT, rel.slice('/out/'.length));
  } else {
    file = path.join(WEB, rel.replace(/^\/+/, ''));
  }
  const relCheck = path.relative(rel.startsWith('/out/') ? OUT : WEB, file);
  if (relCheck.startsWith('..') || path.isAbsolute(relCheck)) return sendError(res, 403, 'forbidden');
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('not found');
      return;
    }
    const ext = path.extname(file).toLowerCase();
    const type = MIME[ext] || 'application/octet-stream';
    const headers = { 'Content-Type': type, 'Cache-Control': 'no-cache' };
    if (ext === '.mp4') {
      // Range support for video playback.
      const range = req.headers.range;
      if (range) {
        const m = /bytes=(\d*)-(\d*)/.exec(range);
        const start = m && m[1] ? Number(m[1]) : 0;
        const end = m && m[2] ? Number(m[2]) : st.size - 1;
        res.writeHead(206, {
          ...headers,
          'Content-Range': `bytes ${start}-${end}/${st.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': end - start + 1,
        });
        fs.createReadStream(file, { start, end }).pipe(res);
        return;
      }
    }
    res.writeHead(200, { ...headers, 'Content-Length': st.size });
    fs.createReadStream(file).pipe(res);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  if (url.pathname.startsWith('/api/')) {
    handleApi(req, res, url).catch((e) => {
      console.error(e);
      sendError(res, 500, String(e && e.message || e));
    });
    return;
  }
  serveStatic(req, res, url);
});

server.listen(PORT, () => {
  console.log(`blur-inversion player: http://localhost:${PORT}`);
  console.log(`workspace root: ${ROOT}`);
});

export { server, CLIPS, getClip };
