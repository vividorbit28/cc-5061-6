// Evaluation stage. This is the ONLY module that touches ground truth, and it
// is called after inversion with truth passed explicitly. It computes:
//   - reconstruction error (PSNR / SSIM) vs the sharp reference latent
//   - forward residual (re-blur the estimate through the assumed model)
//   - ringing index and overshoot near strong edges
//   - per-region error: recoverable chart, unrecoverable fine texture, flat area
//   - noise-floor estimate from the observations themselves (no truth needed)
//
// It is deliberately possible to run inversion without calling any of this.

import { GEOM } from './synth.js';
import { psnr, ssim, rms, median, clamp } from './util.js';

export const REGIONS = {
  // observation coordinates, inclusive-exclusive
  chart: { x0: 96, y0: 20, x1: 128, y1: 45, label: 'bar chart (recoverable detail)' },
  texture: { x0: 27, y0: 48, x1: 59, y1: 80, label: 'fine texture patch (near Nyquist)' },
};

export function regionStats(est, truth, region, w) {
  let se = 0, n = 0, bias = 0;
  const a = [], b = [];
  for (let y = region.y0; y < region.y1; y++) {
    for (let x = region.x0; x < region.x1; x++) {
      const i = y * w + x;
      const d = est[i] - truth[i];
      se += d * d; bias += d; n++;
      a.push(est[i]); b.push(truth[i]);
    }
  }
  const mse = se / n;

  // Correlation of the *detail* (high-pass) in the region. Matching variance is
  // not enough: an estimate can show the same amount of structure yet have it in
  // the wrong places. A correlation near 0 means the visible detail is not the
  // real detail; near 1 means it is.
  const hpEst = [];
  const hpTruth = [];
  for (let y = region.y0; y < region.y1; y++) {
    for (let x = region.x0; x < region.x1; x++) {
      if (x < 1 || y < 1 || x >= w - 1) { hpEst.push(0); hpTruth.push(0); continue; }
      const i = y * w + x;
      hpEst.push(4 * est[i] - est[i - 1] - est[i + 1] - est[i - w] - est[i + w]);
      hpTruth.push(4 * truth[i] - truth[i - 1] - truth[i + 1] - truth[i - w] - truth[i + w]);
    }
  }
  return {
    label: region.label,
    rmse: Math.sqrt(mse),
    psnr: mse > 1e-15 ? 10 * Math.log10(1 / mse) : 99,
    bias: bias / n,
    estStd: std(a),
    truthStd: std(b),
    detailCorr: corr(hpEst, hpTruth),
  };
}

function corr(a, b) {
  let ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
  ma /= a.length; mb /= b.length;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  return saa > 1e-15 && sbb > 1e-15 ? sab / Math.sqrt(saa * sbb) : 0;
}

function std(a) {
  let m = 0;
  for (const v of a) m += v;
  m /= a.length;
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return Math.sqrt(s / a.length);
}

// Find the flattest 24x24 window of the truth to report low-texture behaviour.
export function flattestRegion(truth, w, h, size = 24) {
  let best = null;
  for (let y = 0; y + size <= h; y += 4) {
    for (let x = 0; x + size <= w; x += 4) {
      let s = 0, s2 = 0, n = 0;
      for (let yy = y; yy < y + size; yy++) {
        for (let xx = x; xx < x + size; xx++) {
          const v = truth[yy * w + xx];
          s += v; s2 += v * v; n++;
        }
      }
      const varr = s2 / n - (s / n) ** 2;
      if (!best || varr < best.varr) best = { x0: x, y0: y, x1: x + size, y1: y + size, varr, label: 'flat area (low texture)' };
    }
  }
  return best;
}

// Ringing near strong edges. The error is measured in a 2 px dilation of the
// top-2% gradient pixels of the truth; it is normalised by the mean edge
// contrast so that values are comparable across scenes.
export function ringingIndex(est, truth, w, h) {
  const grad = new Float64Array(w * h);
  const gx = new Float64Array(w * h);
  const gy = new Float64Array(w * h);
  const vals = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = (truth[i + 1] - truth[i - 1]) * 0.5;
      gy[i] = (truth[i + w] - truth[i - w]) * 0.5;
      grad[i] = Math.hypot(gx[i], gy[i]);
      vals.push(grad[i]);
    }
  }
  vals.sort((a, b) => a - b);
  const thr = vals[Math.floor(vals.length * 0.98)];
  const edge = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (grad[i] >= thr && thr > 0) edge[i] = 1;
  // dilate 2
  const dil = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!edge[y * w + x]) continue;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) dil[yy * w + xx] = 1;
        }
      }
    }
  }
  let se = 0, n = 0, contrast = 0, ne = 0, overshoot = 0;
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x;
      if (dil[i]) {
        const d = est[i] - truth[i];
        se += d * d; n++;
      }
      if (edge[i]) {
        let lo = 1, hi = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const v = truth[(y + dy) * w + (x + dx)];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        }
        contrast += hi - lo;
        ne++;
        if (est[i] < lo - 0.04 || est[i] > hi + 0.04) overshoot++;
      }
    }
  }
  const ringRms = n ? Math.sqrt(se / n) : 0;
  const meanContrast = ne ? contrast / ne : 1;
  return {
    index: (100 * ringRms) / Math.max(meanContrast, 1e-6),
    ringRms,
    meanContrast,
    overshootFraction: ne ? overshoot / ne : 0,
    edgePixels: ne,
  };
}

// Noise standard deviation from the observations alone.
//
// Per-block MAD of a Laplacian high-pass, corrected for the filter's noise
// gain. Taking a low percentile across blocks avoids textured regions (which
// inflate the Laplacian response) and sharp edges; the flattest blocks are the
// best available estimate of the sensor noise.
export function estimateNoise(obs, w, h, block = 16) {
  const g = Math.sqrt(20);
  const perBlock = [];
  for (let by = 0; by + block <= h; by += block) {
    for (let bx = 0; bx + block <= w; bx += block) {
      const vals = [];
      for (let y = by; y < by + block; y++) {
        for (let x = bx; x < bx + block; x++) {
          const i = y * w + x;
          if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) continue;
          vals.push(-4 * obs[i] + obs[i - 1] + obs[i + 1] + obs[i - w] + obs[i + w]);
        }
      }
      if (vals.length < 16) continue;
      const m = median(vals);
      const ad = vals.map((v) => Math.abs(v - m));
      perBlock.push((median(ad) * 1.4826) / g);
    }
  }
  if (!perBlock.length) {
    const vals = [];
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      vals.push(-4 * obs[i] + obs[i - 1] + obs[i + 1] + obs[i - w] + obs[i + w]);
    }
    return (median(vals.map((v) => Math.abs(v - median(vals))) ) * 1.4826) / g;
  }
  perBlock.sort((a, b) => a - b);
  const idx = Math.floor(perBlock.length * 0.2);
  return perBlock[Math.min(perBlock.length - 1, idx)];
}

export function autoLambda(sigmaEst) {
  return clamp(0.002 + 9 * Math.pow(Math.max(sigmaEst, 1e-4), 1.15), 0.002, 0.4);
}

export function evaluate(est, truth, reblur, observations, opts = {}) {
  const { obsW, obsH } = GEOM;
  const n = obsW * obsH;
  const sigmaEst = estimateNoise(observations[0], obsW, obsH);
  const err = new Float64Array(n);
  for (let i = 0; i < n; i++) err[i] = est[i] - truth[i];
  let se = 0;
  for (let i = 0; i < n; i++) se += err[i] * err[i];
  const rmse = Math.sqrt(se / n);
  const estimatePsnr = 10 * Math.log10(1 / Math.max(se / n, 1e-18));
  const blurPsnr = psnr(observations[Math.floor(observations.length / 2)], truth);
  const gainDb = estimatePsnr - blurPsnr;

  const flat = flattestRegion(truth, obsW, obsH);
  const regions = [
    regionStats(est, truth, REGIONS.chart, obsW),
    regionStats(est, truth, REGIONS.texture, obsW),
    regionStats(est, truth, flat, obsW),
  ];
  const ring = ringingIndex(est, truth, obsW, obsH);

  // forward residual from the supplied reblur result
  let fres = 0, fn = 0;
  const m = 4;
  for (const r of reblur) {
    for (let y = m; y < obsH - m; y++) {
      for (let x = m; x < obsW - m; x++) {
        const v = r.residual[y * obsW + x];
        fres += v * v; fn++;
      }
    }
  }
  const forwardRmse = Math.sqrt(fres / fn);

  return {
    psnr: estimatePsnr,
    ssim: ssim(est, truth, obsW, obsH),
    rmse,
    gainDb,
    blurPsnr,
    forwardRmse,
    noiseSigmaEstimate: sigmaEst,
    forwardToNoise: forwardRmse / Math.max(sigmaEst, 1e-6),
    ringing: ring,
    regions,
    flatRegion: flat,
    supportCoverage: opts.support && opts.support.texMask
      ? coverage(opts.support.map, opts.support.texMask)
      : null,
  };
}

function coverage(map, mask) {
  let s = 0, n = 0;
  for (let i = 0; i < map.length; i++) {
    if (mask[i]) { s += map[i]; n++; }
  }
  return n ? s / n : 0;
}

export { psnr, ssim };
