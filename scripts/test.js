// Kernel tests. Run with: npm test
// These check mathematical properties, not visual impressions.

import { GEOM, makeClip, forwardFromLatent } from '../src/synth.js';
import { invert, residualStats, compareHypotheses, analyzeClip } from '../src/pipeline.js';
import { evaluate, estimateNoise, autoLambda } from '../src/evaluate.js';
import { psnr, ssim } from '../src/util.js';
import { phaseCorrelate } from '../src/motion.js';
import {
  makeLayeredClip, invertLayered,
  invertSingleLayer, layeredResidualRmse, compareLayeredStrategies,
  estimateForegroundRegion, MAX_LAYERS,
} from '../src/layers.js';
import { embedObservation } from '../src/model.js';

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
console.log('layered motion inversion');
// ---------------------------------------------------------------------------

const embed = (m) => embedObservation(m, obsW, obsH, GEOM.padW, GEOM.padH, GEOM.ox, GEOM.oy);
const oracleSpec = (clip, overrides = {}) => ({
  layers: [
    { kind: 'linear', vx: clip.truthVelocities.bg.vx, vy: clip.truthVelocities.bg.vy, maskCanvas: embed(clip.truthMasks.bg), fixedCoverage: true },
    { kind: 'linear', vx: clip.truthVelocities.fg.vx, vy: clip.truthVelocities.fg.vy, maskCanvas: embed(clip.truthMasks.fg) },
  ],
  refIndex: clip.refIndex, rho: clip.spec.rho, lambda: 0.004, maxIter: 80, tol: 1e-9,
  alternations: 0, estimateMasks: false, ...overrides,
});
const isIterable = (v) => Array.isArray(v) || ArrayBuffer.isView(v);
const flatten = (v) => (isIterable(v) ? Array.from(v).flatMap(flatten) : [v]);
const allFinite = (arrays) => flatten(arrays).every(Number.isFinite);

test('generator coverage is a complementary alpha partition (bg + fg = 1)', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, boundaryMode: 'frozen', noiseSigma: 0, quantize: false });
  for (let i = 0; i < obsW * obsH; i++) approx(clip.truthMasks.bg[i] + clip.truthMasks.fg[i], 1, 1e-9, 'coverage partition');
});

test('frozen-boundary oracle closes the forward loop (re-blur RMSE ~ 0)', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0, quantize: false, boundaryMode: 'frozen' });
  const r = invertLayered(clip.obs, oracleSpec(clip));
  less(layeredResidualRmse(r, clip.obs), 2e-3, 'frozen closed-loop RMSE');
  greater(psnr(r.estimate, clip.clean[clip.refIndex]), 30, 'oracle composite PSNR');
});

test('exact time-sampled boundaries: frozen approximation error concentrates at layer edges', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0, quantize: false, boundaryMode: 'exact' });
  const r = invertLayered(clip.obs, oracleSpec(clip, { maxIter: 40 }));
  const fg = clip.truthMasks.fg;
  let seB = 0, nB = 0, seI = 0, nI = 0;
  for (let k = 0; k < clip.obs.length; k++) {
    for (let y = 2; y < obsH - 2; y++) for (let x = 2; x < obsW - 2; x++) {
      const i = y * obsW + x;
      const d = r.reblur[k].residual[i];
      if (fg[i] > 0.02 && fg[i] < 0.98) { seB += d * d; nB++; }
      else { seI += d * d; nI++; }
    }
  }
  const rb = Math.sqrt(seB / nB), ri = Math.sqrt(seI / nI);
  greater(nB, 0, 'boundary pixels exist');
  greater(rb / ri, 1.25, `boundary/interior residual ratio ${(rb / ri).toFixed(2)}`);
});

test('different-speed fg/bg: layered model beats the single-motion model on residual and PSNR', () => {
  // Frozen-boundary clip shares the inversion's declared approximation, so the
  // comparison isolates the benefit of a second motion layer (the exact clip's
  // boundary-approximation error is tested separately above).
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen' });
  const layered = invertLayered(clip.obs, oracleSpec(clip, { maxIter: 45 }));
  const single = invertSingleLayer(clip.obs, { vx: clip.truthVelocities.bg.vx, vy: clip.truthVelocities.bg.vy, refIndex: clip.refIndex, rho: clip.spec.rho, lambda: 0.004, maxIter: 45 });
  const rL = layeredResidualRmse(layered, clip.obs);
  const rS = layeredResidualRmse(single, clip.obs);
  less(rL, rS, `layered RMSE ${rL.toFixed(4)} vs single ${rS.toFixed(4)}`);
  greater(rS / rL, 4, `forward-residual reduction factor ${(rS / rL).toFixed(1)}x`);
  greater(psnr(layered.estimate, clip.clean[clip.refIndex]) - psnr(single.estimate, clip.clean[clip.refIndex]), 4, 'layered PSNR gain');
});

test('moving foreground over a static background is recovered with closed-loop residual', () => {
  const clip = makeLayeredClip({
    seed: 31, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen',
    bgMotion: { type: 'pan', params: { vx: 0, vy: 0 } }, fgMotion: { vx: -4.5, vy: 1.2 },
  });
  const r = invertLayered(clip.obs, oracleSpec(clip, { maxIter: 60 }));
  less(layeredResidualRmse(r, clip.obs), 8e-3, 'static-bg closed-loop RMSE');
  approx(r.kernels.length, 2, 0, 'one independent kernel per layer');
});

test('layer count is capped at MAX_LAYERS so noise cannot be fit by adding kernels', () => {
  const dummy = () => ({ kind: 'linear', vx: 0, vy: 0, maskCanvas: new Float64Array(GEOM.padW * GEOM.padH) });
  let threw = false;
  try {
    invertLayered([new Float64Array(obsW * obsH)], { layers: Array.from({ length: MAX_LAYERS + 1 }, dummy), estimateMasks: false, maxIter: 1 });
  } catch (e) { threw = /at most/.test(e.message); }
  if (!threw) throw new Error('expected layer-cap error');
});

test('an unidentifiable (zero-coverage) layer is flagged by its uncertainty map, not invented detail', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 3, refIndex: 1, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen' });
  const zero = new Float64Array(GEOM.padW * GEOM.padH);
  const r = invertLayered(clip.obs, {
    layers: [
      { kind: 'linear', vx: clip.truthVelocities.bg.vx, vy: clip.truthVelocities.bg.vy, maskCanvas: embed(clip.truthMasks.bg), fixedCoverage: true },
      { kind: 'linear', vx: 3, vy: 1, maskCanvas: zero },
    ],
    refIndex: 1, rho: 0.8, lambda: 0.01, maxIter: 20, alternations: 0, estimateMasks: false,
  });
  approx(r.uncertainty.length, 2, 0, 'uncertainty map per layer');
  const meanU = r.uncertainty[1].reduce((a, b) => a + b, 0) / r.uncertainty[1].length;
  greater(meanU, 0.9, `zero-coverage layer uncertainty mean ${meanU.toFixed(2)}`);
  const meanX = r.layerEstimates[1].reduce((a, b) => a + Math.abs(b), 0) / r.layerEstimates[1].length;
  less(meanX, 0.05, 'unidentifiable layer carries no invented structure');
});

test('global single-motion inversion runs as the one-layer special config', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen' });
  const r = invertSingleLayer(clip.obs, { vx: clip.truthVelocities.bg.vx, vy: clip.truthVelocities.bg.vy, refIndex: clip.refIndex, rho: clip.spec.rho, lambda: 0.01, maxIter: 30 });
  approx(r.layers.length, 1, 0, 'single layer');
  let cover = 0;
  for (const v of r.layerMasks[0]) cover += v;
  approx(cover / r.layerMasks[0].length, 1, 1e-9, 'full-window coverage');
  less(layeredResidualRmse(r, clip.obs), 0.1, 'single-layer forward residual finite and small-ish');
});

test('data-driven lock region is derived from observations (no hand mask) and detects the moving object', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen' });
  const fg = estimateForegroundRegion(clip.obs, { x: clip.truthVelocities.bg.vx, y: clip.truthVelocities.bg.vy }, 2, obsW, obsH);
  let mx = 0; for (const v of fg) mx = Math.max(mx, v);
  greater(mx, 0.5, 'foreground lock strength');
  const r = invertLayered(clip.obs, {
    layers: [
      { kind: 'linear', vx: clip.truthVelocities.bg.vx, vy: clip.truthVelocities.bg.vy },
      { kind: 'linear', vx: clip.truthVelocities.fg.vx, vy: clip.truthVelocities.fg.vy },
    ],
    refIndex: 2, rho: 0.8, lambda: 0.01, maxIter: 25, alternations: 0, estimateMasks: true,
    bgVelocity: { x: clip.truthVelocities.bg.vx, y: clip.truthVelocities.bg.vy },
  });
  if (!allFinite([r.estimate, ...r.layerEstimates, ...r.uncertainty, ...r.reblur.map((b) => b.residual)])) {
    throw new Error('non-finite value in data-driven inversion');
  }
});

test('strategy comparison scores both parameterisations on the same forward residual', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0, quantize: false, boundaryMode: 'frozen' });
  const cmp = compareLayeredStrategies(clip.obs, oracleSpec(clip, { maxIter: 40 }));
  if (!['independent', 'shared-bg'].includes(cmp.selected)) throw new Error(`bad selection ${cmp.selected}`);
  if (!(cmp.scores.independent > 0 && cmp.scores['shared-bg'] > 0)) throw new Error('scores must be positive finite numbers');
  approx(cmp.scores.independent, layeredResidualRmse(cmp.results.independent, clip.obs), 1e-12, 'independent score consistency');
  // On the genuinely two-motion frozen clip the joint independent fit wins.
  if (cmp.selected !== 'independent') throw new Error(`expected independent, got ${cmp.selected}`);
});

test('layered outputs are finite: estimates, masks, kernels, uncertainty and residuals', () => {
  const clip = makeLayeredClip({ seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.003, quantize: true, boundaryMode: 'exact' });
  const r = invertLayered(clip.obs, oracleSpec(clip, { maxIter: 30 }));
  if (!allFinite([r.estimate, ...r.layerEstimates, ...r.layerMasks, ...r.uncertainty, ...r.coverage, ...r.reblur.map((b) => b.predicted), ...r.reblur.map((b) => b.residual), ...r.kernels.map((k) => k.data)])) {
    throw new Error('NaN/Inf escaped the layered solver');
  }
});

// ---------------------------------------------------------------------------
console.log('');
console.log(results.join('\n'));
console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
