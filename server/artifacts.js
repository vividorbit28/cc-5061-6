// Offline artifact builder: renders the deliverable MP4s, PNG comparison sheets,
// and a metrics report. Uses the same invert() path as the web UI.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { GEOM, makeClip, MOTION_PRESETS, DEFAULTS } from '../src/synth.js';
import { invert, residualStats, compareHypotheses, analyzeClip } from '../src/pipeline.js';
import { evaluate, autoLambda, estimateNoise } from '../src/evaluate.js';
import { psnr, ssim, clamp } from '../src/util.js';
import { encodePNG, encodePNGColor, encodeMP4, drawText, drawRect } from './media.js';
import { turbo } from '../src/util.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'out');
const { obsW, obsH } = GEOM;

export const SIM_CASES = [
  {
    id: 'pan-clean',
    title: 'Pan / clean / short exposure',
    clip: { seed: 3, motionType: 'pan', exposureKind: 'uniform', rho: 0.75, noiseSigma: 0, quantize: false },
    assumptionFrom: null, // use the preset params
    lambdaMode: 'auto',
  },
  {
    id: 'pan-long',
    title: 'Pan / long exposure / centered shutter',
    clip: { seed: 3, motionType: 'pan', exposureKind: 'centered', rho: 0.95, noiseSigma: 0.003, quantize: true },
    lambdaMode: 'auto',
  },
  {
    id: 'diag',
    title: 'Diagonal pan / ramp shutter',
    clip: { seed: 7, motionType: 'diag', exposureKind: 'ramp', rho: 0.8, noiseSigma: 0.003, quantize: true },
    lambdaMode: 'auto',
  },
  {
    id: 'fast',
    title: 'Fast pan / strong blur',
    clip: { seed: 11, motionType: 'fast', exposureKind: 'uniform', rho: 0.9, noiseSigma: 0.003, quantize: true },
    lambdaMode: 'auto',
  },
  {
    id: 'shake',
    title: 'Pan + shake / non-constant velocity',
    clip: { seed: 5, motionType: 'shake', exposureKind: 'uniform', rho: 0.8, noiseSigma: 0.003, quantize: true },
    lambdaMode: 'auto',
  },
  {
    id: 'noisy',
    title: 'Pan / high noise (sigma 0.012)',
    clip: { seed: 17, motionType: 'pan', exposureKind: 'uniform', rho: 0.85, noiseSigma: 0.012, quantize: true },
    lambdaMode: 'auto',
  },
];

function assumptionFor(spec, clip) {
  const p = clip.motion.params;
  const sigma = estimateNoise(clip.obs[0], obsW, obsH);
  return {
    kind: spec.motionType === 'shake' ? 'shake' : spec.motionType === 'arc' ? 'arc' : 'linear',
    vx: p.vx ?? 0, vy: p.vy ?? 0, amp: p.amp ?? 0, freq: p.freq ?? 0,
    omega: p.omega ?? 0, pivotX: p.pivotX ?? 0, pivotY: p.pivotY ?? 0,
    rho: spec.rho, exposureKind: spec.exposureKind,
    refIndex: clip.refIndex, lambda: autoLambda(sigma), tv: 0, maxIter: 55,
  };
}

// Side-by-side comparison frame: blurred | estimate | truth | residual.
// Images are upscaled 2x with nearest-neighbour and separated by gutters.
export function composeComparison(obsImg, estImg, truthImg, resImg, w, h, opts = {}) {
  const scale = opts.scale ?? 2;
  const pad = 8;
  const labelH = 18;
  const tiles = [
    { img: obsImg, label: 'BLURRED INPUT' },
    { img: estImg, label: 'INVERTED ESTIMATE' },
    { img: truthImg, label: 'GROUND TRUTH (EVAL ONLY)' },
    { img: resImg, label: (opts.residualLabel || 'FORWARD RESIDUAL') },
  ];
  const tw = w * scale;
  const th = h * scale;
  const ow = tiles.length * tw + (tiles.length + 1) * pad;
  const oh = th + labelH + 2 * pad;
  const rgb = new Uint8Array(ow * oh * 3);
  for (let i = 0; i < ow * oh; i++) { rgb[3 * i] = 14; rgb[3 * i + 1] = 14; rgb[3 * i + 2] = 18; }
  tiles.forEach((t, ti) => {
    const x0 = pad + ti * (tw + pad);
    const y0 = pad + labelH;
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const v = Math.round(clamp(t.img[((y / scale) | 0) * w + ((x / scale) | 0)]) * 255);
        const o = ((y0 + y) * ow + x0 + x) * 3;
        rgb[o] = v; rgb[o + 1] = v; rgb[o + 2] = v;
      }
    }
    drawTextRGB(rgb, ow, oh, t.label, x0 + 2, pad + 2, 1, [220, 220, 230]);
  });
  return { rgb, w: ow, h: oh };
}

function drawTextRGB(rgb, w, h, text, x0, y0, scale, color) {
  // Reuse the grayscale font by drawing onto a scratch mask.
  const mask = new Float64Array(w * h);
  drawText(mask, w, h, text, x0, y0, scale, 1, 0);
  for (let i = 0; i < w * h; i++) {
    if (mask[i] > 0) {
      rgb[3 * i] = color[0];
      rgb[3 * i + 1] = color[1];
      rgb[3 * i + 2] = color[2];
    }
  }
}

// Stretch the support map between its 5th and 95th percentile for legibility.
function stretchSupport(support) {
  const sorted = Array.from(support).sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
  const hi = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
  const span = Math.max(hi - lo, 1e-3);
  return Float64Array.from(support, (v) => Math.max(0, Math.min(1, (v - lo) / span)));
}

function residualVis(residual, gain = 5) {
  // Auto-scale to the 99th percentile so a well-fit residual is still legible.
  const mags = Array.from(residual, (v) => Math.abs(v)).sort((a, b) => a - b);
  const p99 = mags[Math.min(mags.length - 1, Math.floor(mags.length * 0.99))] || 1e-6;
  const g = gain ?? 1 / Math.max(p99, 1e-6);
  return residual.map((v) => clamp(Math.abs(v) * g));
}

function residualAutoLabel(residual) {
  const mags = Array.from(residual, (v) => Math.abs(v)).sort((a, b) => a - b);
  const p99 = mags[Math.min(mags.length - 1, Math.floor(mags.length * 0.99))] || 1e-6;
  return { p99, gain: 1 / Math.max(p99, 1e-6) };
}

function heatRGB(img, maxVal) {
  const out = new Uint8Array(img.length * 3);
  const inv = maxVal > 0 ? 1 / maxVal : 0;
  for (let i = 0; i < img.length; i++) {
    const [r, g, b] = turbo(clamp(img[i] * inv));
    out[3 * i] = Math.round(r * 255);
    out[3 * i + 1] = Math.round(g * 255);
    out[3 * i + 2] = Math.round(b * 255);
  }
  return out;
}

export async function buildArtifacts({ only = null, onProgress = null } = {}) {
  fs.mkdirSync(OUT, { recursive: true });
  const report = [];
  const jobs = only ? SIM_CASES.filter((c) => only.includes(c.id)) : SIM_CASES;

  let caseIndex = 0;
  for (const spec of jobs) {
    caseIndex++;
    onProgress?.('inverting', `${spec.id} (${caseIndex}/${jobs.length})`);
    const clip = makeClip({ ...DEFAULTS, frames: 7, refIndex: 3, ...spec.clip });
    const assumption = assumptionFor(spec.clip, clip);
    const t0 = performance.now();
    const main = invert(clip.obs, assumption, {});
    const dt = performance.now() - t0;
    const ev = evaluate(main.estimate, clip.refSharp, main.reblur, clip.obs, {});
    const st = residualStats(main.reblur, obsW, obsH);

    // Wrong-motion control: same solve, reversed direction.
    const rev = invert(clip.obs, { ...assumption, vx: -assumption.vx, vy: -assumption.vy }, { computeSupport: false });
    const revEv = evaluate(rev.estimate, clip.refSharp, rev.reblur, clip.obs, {});
    const revSt = residualStats(rev.reblur, obsW, obsH);

    const entry = {
      id: spec.id, title: spec.title,
      assumption, iterations: main.iterations, solveMs: Math.round(dt),
      psnr: ev.psnr, ssim: ev.ssim, blurPsnr: ev.blurPsnr, gainDb: ev.gainDb,
      forwardRmse: ev.forwardRmse, noiseSigma: ev.noiseSigmaEstimate,
      forwardToNoise: ev.forwardToNoise,
      ringing: ev.ringing, regions: ev.regions,
      reversed: { psnr: revEv.psnr, forwardRmse: revSt.rmse },
      history: main.history,
    };
    report.push(entry);

    // Static artifacts
    fs.writeFileSync(path.join(OUT, `${spec.id}_estimate.png`), encodePNG(main.estimate, obsW, obsH));
    fs.writeFileSync(path.join(OUT, `${spec.id}_blurred_ref.png`), encodePNG(clip.obs[clip.refIndex], obsW, obsH));
    fs.writeFileSync(path.join(OUT, `${spec.id}_truth.png`), encodePNG(clip.refSharp, obsW, obsH));
    fs.writeFileSync(path.join(OUT, `${spec.id}_support.png`), encodePNG(stretchSupport(main.support), obsW, obsH));
    fs.writeFileSync(path.join(OUT, `${spec.id}_residual.png`),
      encodePNGColor(heatRGB(combinedResidual(main.reblur), 0.06), obsW, obsH));
    const autoRes = residualAutoLabel(combinedResidual(main.reblur));
    const comp = composeComparison(
      clip.obs[clip.refIndex], main.estimate, clip.refSharp,
      residualVis(combinedResidual(main.reblur), autoRes.gain), obsW, obsH, { scale: 2 },
      { residualLabel: `FORWARD RESIDUAL x${autoRes.gain.toFixed(0)}` },
    );
    fs.writeFileSync(path.join(OUT, `${spec.id}_comparison.png`),
      encodePNGColor(comp.rgb, comp.w, comp.h));

    let panelW = obsW, panelH = obsH;
    // Video: four-panel sequence over the clip frames.
    //   clean (truth, eval only) | blurred input | inverted estimate | residual
    const fps = 6;
    const frames = [];
    const nClip = clip.obs.length;
    const scale = 2;
    for (let k = 0; k < nClip; k++) {
      const resK = main.reblur[k];
      const auto = residualAutoLabel(resK.residual);
      const resVis = residualVis(resK.residual, auto.gain);
      const panel = composePanels([
        { img: clip.clean[k], label: 'CLEAN SEQUENCE (TRUTH, EVAL ONLY)' },
        { img: clip.obs[k], label: 'BLURRED INPUT' },
        { img: main.estimate, label: 'INVERTED ESTIMATE' },
        { img: resVis, label: `FORWARD RESIDUAL x${auto.gain.toFixed(0)} (auto)` },
      ], obsW, obsH, scale);
      // composite into grayscale frame by averaging the RGB channels
      const g = new Float64Array(panel.w * panel.h);
      for (let i = 0; i < g.length; i++) g[i] = (panel.rgb[3 * i] + panel.rgb[3 * i + 1] + panel.rgb[3 * i + 2]) / (3 * 255);
      frames.push(g);
      panelW = panel.w; panelH = panel.h;
    }
    onProgress?.('encoding video', `${spec.id} (${caseIndex}/${jobs.length})`);
    const vres = await encodeMP4(frames, panelW, panelH, path.join(OUT, `${spec.id}.mp4`), { fps });
    entry.video = path.basename(vres.path ?? `${spec.id}.mp4`);
    entry.videoOk = vres.ok;
  }

  fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({
    generatedAt: new Date().toISOString(),
    geometry: GEOM,
    note: 'Metrics are from synthetic data. Ground truth is used for scoring only; inversion receives observations and the stated assumptions.',
    cases: report,
  }, null, 2));

  // Markdown summary.
  const lines = [];
  lines.push('# Blur inversion — synthetic results');
  lines.push('');
  lines.push('Ground truth is synthetic and used for scoring only. The solver receives the blurred observations and the stated motion/shutter assumptions.');
  lines.push('');
  lines.push('| clip | blur PSNR | estimate PSNR | gain | SSIM | forward RMSE | noise σ | fwd/σ | ringing | reversed fwd RMSE |');
  lines.push('|---|---|---|---|---|---|---|---|---|---|');
  for (const r of report) {
    lines.push(`| ${r.id} | ${r.blurPsnr.toFixed(2)} | ${r.psnr.toFixed(2)} | +${r.gainDb.toFixed(2)} dB | ${r.ssim.toFixed(3)} | ${r.forwardRmse.toFixed(4)} | ${r.noiseSigma.toFixed(4)} | ${r.forwardToNoise.toFixed(2)} | ${r.ringing.index.toFixed(1)} | ${r.reversed.forwardRmse.toFixed(4)} |`);
  }
  lines.push('');
  lines.push('`fwd/σ` near 1 means the re-blurred estimate explains the input down to the noise floor; well above 1 means the assumed model does not explain the data.');
  lines.push('');
  lines.push('Per-region detail: `std` is the standard deviation of the estimate vs the truth in that region; `detail r` is the correlation of high-pass content. Matching variance alone can be faked by noise, so the correlation is the honest test of whether detail was recovered.');
  lines.push('');
  lines.push('| clip | region | RMSE | PSNR | std est/truth | detail r |');
  lines.push('|---|---|---|---|---|---|');
  for (const r of report) {
    for (const g of r.regions) {
      lines.push(`| ${r.id} | ${g.label} | ${g.rmse.toFixed(4)} | ${g.psnr.toFixed(1)} | ${g.estStd.toFixed(3)} / ${g.truthStd.toFixed(3)} | ${g.detailCorr == null ? '–' : g.detailCorr.toFixed(2)} |`);
    }
  }
  fs.writeFileSync(path.join(OUT, 'report.md'), lines.join('\n'));

  return report;
}

// Compose labelled panels into a single RGB frame.
export function composePanels(panels, w, h, scale = 2) {
  const pad = 8;
  const labelH = 14;
  const tw = w * scale;
  const th = h * scale;
  const ow = panels.length * tw + (panels.length + 1) * pad;
  const oh = th + labelH + 2 * pad;
  const rgb = new Uint8Array(ow * oh * 3);
  for (let i = 0; i < ow * oh; i++) { rgb[3 * i] = 12; rgb[3 * i + 1] = 14; rgb[3 * i + 2] = 20; }
  panels.forEach((p, pi) => {
    const x0 = pad + pi * (tw + pad);
    const y0 = pad + labelH;
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const v = Math.round(clamp(p.img[((y / scale) | 0) * w + ((x / scale) | 0)]) * 255);
        const o = ((y0 + y) * ow + x0 + x) * 3;
        rgb[o] = v; rgb[o + 1] = v; rgb[o + 2] = v;
      }
    }
    drawTextRGB(rgb, ow, oh, p.label, x0 + 2, pad + 1, 1, [210, 220, 235]);
  });
  return { rgb, w: ow, h: oh };
}

function combinedResidual(reblur) {
  const acc = new Float64Array(obsW * obsH);
  for (const r of reblur) for (let i = 0; i < acc.length; i++) acc[i] += Math.abs(r.residual[i]);
  for (let i = 0; i < acc.length; i++) acc[i] /= reblur.length;
  return acc;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const onlyArg = process.argv.find((a) => a.startsWith('--only='));
  const only = onlyArg ? onlyArg.split('=')[1].split(',') : null;
  buildArtifacts({ only }).then((r) => {
    console.log(`built ${r.length} cases into ${OUT}`);
  }).catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
