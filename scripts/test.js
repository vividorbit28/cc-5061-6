// Kernel tests. Run with: npm test
// These check mathematical properties, not visual impressions.

import { GEOM, makeClip, forwardFromLatent } from '../src/synth.js';
import { invert, residualStats, compareHypotheses, analyzeClip } from '../src/pipeline.js';
import { evaluate, estimateNoise, autoLambda } from '../src/evaluate.js';
import { psnr, ssim, rms } from '../src/util.js';
import { phaseCorrelate } from '../src/motion.js';

let passed = 0;
let failed = 0;
const results = [];

function test(name, fn) {
  try {
    const r = fn();
    if (r === false) throw new Error('returned false');
    passed++;
    results.push(`  ok   ${name}`);
  } catch (e) {
    failed++;
    results.push(`  FAIL ${name}\n       ${e.message}`);
  }
}

function approx(a, b, tol, msg) {
  if (!(Math.abs(a - b) <= tol)) throw new Error(`${msg || 'approx'}: ${a} vs ${b} (tol ${tol})`);
}
function less(a, b, msg) {
  if (!(a < b)) throw new Error(`${msg || 'less'}: ${a} !< ${b}`);
}
function greater(a, b, msg) {
  if (!(a > b)) throw new Error(`${msg || 'greater'}: ${a} !> ${b}`);
}
const { obsW, obsH } = GEOM;

// ---------------------------------------------------------------------------
console.log('forward-model consistency');
// ---------------------------------------------------------------------------

test('generator and inversion share one discrete operator (zero residual)', () => {
  const clip = makeClip({ seed: 3, motionType: 'pan', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.8, noiseSigma: 0, quantize: false });
  const pred = forwardFromLatent(clip.latent, clip.truthOperators);
  let maxErr = 0;
  for (let k = 0; k < pred.length; k++) {
    for (let i = 0; i < pred[k].length; i++) maxErr = Math.max(maxErr, Math.abs(pred[k][i] - clip.obs[k][i]));
  }
  approx(maxErr, 0, 1e-12, 'truth-forward mismatch');
});

test('phase correlation recovers integer/fractional shifts with correct sign', () => {
  const w = 64, h = 64;
  const a = new Float64Array(w * h);
  let s = 12345;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let i = 0; i < a.length; i++) a[i] = rnd();
  const shift = (img, dx, dy) => {
    const out = new Float64Array(w * h);
    const g = (x, y) => img[((y % h + h) % h) * w + ((x % w + w) % w)];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const sx = x - dx, sy = y - dy;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const tx = sx - x0, ty = sy - y0;
      out[y * w + x] = g(x0, y0) * (1 - tx) * (1 - ty) + g(x0 + 1, y0) * tx * (1 - ty)
        + g(x0, y0 + 1) * (1 - tx) * ty + g(x0 + 1, y0 + 1) * tx * ty;
    }
    return out;
  };
  const r = phaseCorrelate(a, shift(a, 3, -4), w, h);
  approx(r.dx, 3, 0.35, 'dx');
  approx(r.dy, -4, 0.35, 'dy');
});

// ---------------------------------------------------------------------------
console.log('inversion quality');
// ---------------------------------------------------------------------------

test('clean case: estimate PSNR beats blurred input by >6 dB', () => {
  const clip = makeClip({ seed: 3, motionType: 'pan', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.75, noiseSigma: 0, quantize: false });
  const r = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.75, exposureKind: 'uniform', refIndex: 3, lambda: 0.005, maxIter: 40 });
  const p = psnr(r.estimate, clip.refSharp);
  const pb = psnr(clip.obs[3], clip.refSharp);
  greater(p - pb, 6, `estimate ${p.toFixed(2)} vs blur ${pb.toFixed(2)}`);
});

test('forward residual of the correct model is at the noise floor', () => {
  const clip = makeClip({ seed: 7, motionType: 'diag', exposureKind: 'ramp', frames: 7, refIndex: 3, rho: 0.8, noiseSigma: 0.004, quantize: true });
  const sigma = estimateNoise(clip.obs[0], obsW, obsH);
  const lam = autoLambda(sigma);
  const r = invert(clip.obs, { vx: 6.5, vy: 3.8, rho: 0.8, exposureKind: 'ramp', refIndex: 3, lambda: lam, maxIter: 45 });
  const st = residualStats(r.reblur, obsW, obsH);
  less(st.rmse, sigma * 2.0, `forward residual ${st.rmse.toFixed(4)} vs sigma ${sigma.toFixed(4)}`);
});

test('wrong motion hypothesis produces a much larger forward residual', () => {
  const clip = makeClip({ seed: 3, motionType: 'diag', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.8, noiseSigma: 0.003, quantize: true });
  const good = invert(clip.obs, { vx: 6.5, vy: 3.8, rho: 0.8, exposureKind: 'uniform', refIndex: 3, lambda: 0.02, maxIter: 35 });
  const bad = invert(clip.obs, { vx: -6.5, vy: -3.8, rho: 0.8, exposureKind: 'uniform', refIndex: 3, lambda: 0.02, maxIter: 35 });
  const sg = residualStats(good.reblur, obsW, obsH).rmse;
  const sb = residualStats(bad.reblur, obsW, obsH).rmse;
  greater(sb / sg, 5, `reversed/good forward ratio ${(sb / sg).toFixed(1)}`);
});

test('hypothesis comparison ranks the true velocity first', () => {
  const clip = makeClip({ seed: 5, motionType: 'pan', exposureKind: 'centered', frames: 7, refIndex: 3, rho: 0.8, noiseSigma: 0.003, quantize: true });
  const cmp = compareHypotheses(clip.obs, [
    { id: 'truth', vx: 7.0, vy: 1.2, rho: 0.8, exposureKind: 'centered' },
    { id: 'wrong', vx: 3.4, vy: 0.6, rho: 0.8, exposureKind: 'centered' },
    { id: 'stationary', vx: 0, vy: 0, rho: 0.8, exposureKind: 'centered' },
  ]);
  if (cmp.bestId !== 'truth') throw new Error(`best = ${cmp.bestId}, expected truth`);
  greater(cmp.margin, 0.2, 'margin');
});

test('more frames improve reconstruction under noise (1 vs 7)', () => {
  const base = { seed: 11, motionType: 'diag', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.8, noiseSigma: 0.005, quantize: true };
  const clip = makeClip(base);
  const keepSingle = [clip.obs[3]];
  const r1 = invert(keepSingle, { vx: 6.5, vy: 3.8, rho: 0.8, exposureKind: 'uniform', refIndex: 0, lambda: 0.03, maxIter: 40 });
  const r7 = invert(clip.obs, { vx: 6.5, vy: 3.8, rho: 0.8, exposureKind: 'uniform', refIndex: 3, lambda: 0.03, maxIter: 40 });
  const p1 = psnr(r1.estimate, clip.refSharp);
  const p7 = psnr(r7.estimate, clip.refSharp);
  greater(p7, p1 + 0.8, `7-frame ${p7.toFixed(2)} vs 1-frame ${p1.toFixed(2)}`);
});

// ---------------------------------------------------------------------------
console.log('noise handling');
// ---------------------------------------------------------------------------

test('noise estimator tracks the true sigma within 25%', () => {
  for (const sigma of [0.003, 0.008, 0.015]) {
    const clip = makeClip({ seed: 3, motionType: 'pan', exposureKind: 'uniform', frames: 3, refIndex: 1, rho: 0.8, noiseSigma: sigma, quantize: true });
    const est = estimateNoise(clip.obs[0], obsW, obsH);
    const rel = Math.abs(est - sigma) / sigma;
    if (rel > 0.25) throw new Error(`sigma ${sigma}: estimate ${est.toFixed(4)} (${(rel * 100).toFixed(0)}% off)`);
  }
});

test('auto lambda increases monotonically with noise', () => {
  const l1 = autoLambda(0.002);
  const l2 = autoLambda(0.006);
  const l3 = autoLambda(0.015);
  less(l1, l2, 'lambda monotone 1->2');
  less(l2, l3, 'lambda monotone 2->3');
});

test('noisy data: stronger lambda reduces error at high noise', () => {
  const clip = makeClip({ seed: 17, motionType: 'pan', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.85, noiseSigma: 0.012, quantize: true });
  const low = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.85, exposureKind: 'uniform', refIndex: 3, lambda: 0.003, maxIter: 40 });
  const high = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.85, exposureKind: 'uniform', refIndex: 3, lambda: 0.08, maxIter: 40 });
  const pl = psnr(low.estimate, clip.refSharp);
  const ph = psnr(high.estimate, clip.refSharp);
  greater(ph, pl + 1, `high-lambda ${ph.toFixed(2)} vs low ${pl.toFixed(2)}`);
});

// ---------------------------------------------------------------------------
console.log('prior vs data separation');
// ---------------------------------------------------------------------------

test('fine texture region is NOT recovered (est std << truth std)', () => {
  const clip = makeClip({ seed: 7, motionType: 'fast', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.9, noiseSigma: 0.003, quantize: true });
  const r = invert(clip.obs, { vx: 11, vy: 0.5, rho: 0.9, exposureKind: 'uniform', refIndex: 3, lambda: 0.03, maxIter: 45 });
  const ev = evaluate(r.estimate, clip.refSharp, r.reblur, clip.obs, {});
  const tex = ev.regions.find((x) => x.label.includes('fine texture'));
  less(tex.estStd, tex.truthStd * 0.8, `texture std est ${tex.estStd.toFixed(3)} vs truth ${tex.truthStd.toFixed(3)}`);
});

test('flat region has low error and low artificial variance', () => {
  const clip = makeClip({ seed: 3, motionType: 'pan', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.75, noiseSigma: 0, quantize: false });
  const r = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.75, exposureKind: 'uniform', refIndex: 3, lambda: 0.006, maxIter: 40 });
  const ev = evaluate(r.estimate, clip.refSharp, r.reblur, clip.obs, {});
  const flat = ev.regions.find((x) => x.label.includes('flat'));
  less(flat.rmse, 0.03, 'flat rmse');
  less(flat.estStd, 0.02, 'flat artificial std');
});

test('support map: flat/measured areas score higher than the fine-texture patch', () => {
  const clip = makeClip({ seed: 7, motionType: 'fast', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.9, noiseSigma: 0.003, quantize: true });
  const r = invert(clip.obs, { vx: 11, vy: 0.5, rho: 0.9, exposureKind: 'uniform', refIndex: 3, lambda: 0.03, maxIter: 45 });
  // texture region from evaluate.js: x 27..59, y 48..80
  let tex = 0, nTex = 0, all = 0;
  for (let y = 48; y < 80; y++) for (let x = 27; x < 59; x++) { tex += r.support[y * obsW + x]; nTex++; }
  for (const v of r.support) all += v;
  tex /= nTex; all /= r.support.length;
  less(tex, all, `texture support ${tex.toFixed(3)} vs mean ${all.toFixed(3)}`);
});

// ---------------------------------------------------------------------------
console.log('ambiguity handling');
// ---------------------------------------------------------------------------

test('reversed hypothesis is rejected when inter-frame motion is present', () => {
  const clip = makeClip({ seed: 5, motionType: 'pan', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.8, noiseSigma: 0.003, quantize: true });
  const fwd = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.8, exposureKind: 'uniform', refIndex: 3, lambda: 0.02, maxIter: 35 });
  const rev = invert(clip.obs, { vx: -7, vy: -1.2, rho: 0.8, exposureKind: 'uniform', refIndex: 3, lambda: 0.02, maxIter: 35 });
  const sf = residualStats(fwd.reblur, obsW, obsH).rmse;
  const sr = residualStats(rev.reblur, obsW, obsH).rmse;
  greater(sr / sf, 5, `reversed forward residual ratio ${(sr / sf).toFixed(1)}`);
});

test('stationary frames leave direction unidentifiable (registration ~0)', () => {
  const clip = makeClip({ seed: 13, motionType: 'stationary', exposureKind: 'centered', frames: 7, refIndex: 3, rho: 0.6, noiseSigma: 0.002, quantize: true });
  const a = analyzeClip(clip.obs, { refIndex: 3, rho: 0.6 });
  less(Math.hypot(a.candidate.vx, a.candidate.vy), 0.5, 'registered velocity for stationary clip');
});

test('dead-zone: motion shorter than one pixel is treated as stationary', () => {
  const clip = makeClip({ seed: 3, motionType: 'pan', exposureKind: 'uniform', frames: 5, refIndex: 2, rho: 0.3, noiseSigma: 0, quantize: false, motionParams: { vx: 0.4, vy: 0.2 } });
  const a = analyzeClip(clip.obs, { refIndex: 2, rho: 0.3 });
  less(Math.hypot(a.candidate.vx, a.candidate.vy), 1.0, 'sub-pixel motion should register near zero');
});

// ---------------------------------------------------------------------------
console.log('metrics');
// ---------------------------------------------------------------------------

test('PSNR is monotone in error', () => {
  const a = new Float64Array(100).fill(0.5);
  const b = new Float64Array(100).fill(0.51);
  const c = new Float64Array(100).fill(0.6);
  greater(psnr(a, b), psnr(a, c), 'psnr monotone');
});

test('SSIM is 1 for identical images', () => {
  const a = new Float64Array(64 * 64);
  for (let i = 0; i < a.length; i++) a[i] = (Math.sin(i * 0.1) + 1) / 2;
  approx(ssim(a, a, 64, 64), 1, 1e-6, 'ssim identity');
});

test('ringing grows when an under-regularised solve amplifies noise/edges', () => {
  const clip = makeClip({ seed: 3, motionType: 'pan', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.75, noiseSigma: 0.01, quantize: true });
  const reg = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.75, exposureKind: 'uniform', refIndex: 3, lambda: 0.12, maxIter: 40 });
  const over = invert(clip.obs, { vx: 7, vy: 1.2, rho: 0.75, exposureKind: 'uniform', refIndex: 3, lambda: 0.001, maxIter: 80 });
  const e1 = evaluate(reg.estimate, clip.refSharp, reg.reblur, clip.obs, {});
  const e2 = evaluate(over.estimate, clip.refSharp, over.reblur, clip.obs, {});
  greater(e2.ringing.index, e1.ringing.index, 'ringing should grow when under-regularised');
});

test('estimate is bounded and finite (no NaN/Inf escape)', () => {
  for (const lambda of [0.001, 0.05, 0.2]) {
    const clip = makeClip({ seed: 17, motionType: 'fast', exposureKind: 'uniform', frames: 7, refIndex: 3, rho: 0.9, noiseSigma: 0.012, quantize: true });
    const r = invert(clip.obs, { vx: 11, vy: 0.5, rho: 0.9, exposureKind: 'uniform', refIndex: 3, lambda, maxIter: 40 });
    let mn = Infinity, mx = -Infinity;
    for (const v of r.estimate) {
      if (!Number.isFinite(v)) throw new Error(`non-finite value at lambda ${lambda}`);
      mn = Math.min(mn, v); mx = Math.max(mx, v);
    }
    less(mn, 0.2, 'lower bound');
    greater(mx, 0.6, 'upper bound');
  }
});

// ---------------------------------------------------------------------------
console.log('');
console.log(results.join('\n'));
console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
