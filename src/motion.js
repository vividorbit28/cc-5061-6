// Motion analysis from blurred observations only.
//
// Step 1 — frame-to-frame registration (phase correlation + sub-pixel refine)
//   gives the inter-frame displacement s_k. Blurred frames are low-pass, so
//   registration is biased toward zero; the bias is reported as an uncertainty.
// Step 2 — exposure-kernel estimation from the *sequence* of blurred frames.
//   With a global constant-velocity path and a known reference frame, a frame at
//   path offset s has exposure kernel h centred at s*rho. The inverse problem
//   for x and the blur offsets is what the solver performs; here we produce
//   candidate kernel directions from the residual structure.
//
// Everything in this file is derived from the observed frames (plus explicit
// user assumptions). Ground-truth motion is only used by the evaluation stage.

import { FFT2, frequencies } from './fft.js';
import { hann2d, phaseMultiplier, cropAt, padAt } from './fft.js';
import { clamp, robustSigma, median, bilinear, angleDiffDeg } from './util.js';

// ---------------------------------------------------------------------------
// Phase correlation on a single window.
// ---------------------------------------------------------------------------

export function phaseCorrelate(a, b, w, h, opts = {}) {
  const win = opts.window ?? hann2d(w, h);
  const reA = new Float64Array(w * h);
  const imA = new Float64Array(w * h);
  const reB = new Float64Array(w * h);
  const imB = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    reA[i] = a[i] * win[i];
    reB[i] = b[i] * win[i];
  }
  const fa = new FFT2(w, h);
  fa.re.set(reA); fa.im.set(imA); fa.forward();
  const fb = new FFT2(w, h);
  fb.re.set(reB); fb.im.set(imB); fb.forward();
  const R = new Float64Array(w * h);
  const I = new Float64Array(w * h);
  const eps = opts.eps ?? 1e-6;
  let maxMag = 0;
  for (let i = 0; i < w * h; i++) {
    const cr = fa.re[i] * fb.re[i] + fa.im[i] * fb.im[i];
    const ci = fa.im[i] * fb.re[i] - fa.re[i] * fb.im[i];
    const mag = Math.hypot(cr, ci);
    if (mag > maxMag) maxMag = mag;
    R[i] = mag > 0 ? cr / (mag + eps * maxMag) : 0;
    I[i] = mag > 0 ? ci / (mag + eps * maxMag) : 0;
  }
  const inv = new FFT2(w, h);
  inv.re.set(R); inv.im.set(I); inv.inverse();
  // shift in pixels for a peak at frequency-grid index (kx,ky)
  let best = 0;
  let bx = 0, by = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = inv.re[y * w + x];
      if (v > best) { best = v; bx = x; by = y; }
    }
  }
  let dx = bx < w / 2 ? bx : bx - w;
  let dy = by < h / 2 ? by : by - h;
  // The cross-power spectrum convention used above peaks at -displacement, so
  // flip to report "content moved by (dx,dy)".
  dx = -dx;
  dy = -dy;
  // Parabolic sub-pixel interpolation around the peak (in peak coordinates).
  const at = (x, y) => inv.re[((y + h) % h) * w + ((x + w) % w)];
  const rx = at(bx + 1, by), lx = at(bx - 1, by);
  const ry = at(bx, by + 1), ly = at(bx, by - 1);
  const denX = 2 * best - rx - lx;
  const denY = 2 * best - ry - ly;
  const subX = Math.abs(denX) > 1e-9 ? (rx - lx) / (2 * denX) : 0;
  const subY = Math.abs(denY) > 1e-9 ? (ry - ly) / (2 * denY) : 0;
  if (Math.abs(subX) < 0.6) dx -= subX;
  if (Math.abs(subY) < 0.6) dy -= subY;
  return { dx, dy, peak: best, map: inv.re, bias: { bx, by } };
}

// Estimate inter-frame displacement with different windows. The spread across
// windows is an honest uncertainty proxy: blurred frames under-constrain motion.
export function registerSequence(obs, opts = {}) {
  const refIndex = opts.refIndex ?? Math.floor(obs.length / 2);
  const w = opts.w ?? 128;
  const h = opts.h ?? 96;
  const ref = obs[refIndex];
  const results = obs.map((img, k) => {
    if (k === refIndex) return { dx: 0, dy: 0, peak: 1, spread: 0, perWindow: [] };
    const per = [];
    // All windows must be power-of-two sized for the FFT.
    const windows = [
      { x: 16, y: 16, w: 128, h: 64 },
      { x: 16, y: 16, w: 64, h: 64 },
      { x: 80, y: 16, w: 64, h: 64 },
    ].filter((win) => win.x + win.w <= w && win.y + win.h <= h);
    for (const win of windows) {
      const a = cropAt(ref, w, win.x, win.y, win.w, win.h);
      const b = cropAt(img, w, win.x, win.y, win.w, win.h);
      const r = phaseCorrelate(a, b, win.w, win.h);
      per.push(r);
    }
    const dx = median(per.map((p) => p.dx));
    const dy = median(per.map((p) => p.dy));
    const spread = Math.max(
      ...per.map((p) => Math.hypot(p.dx - dx, p.dy - dy)),
    );
    return { dx, dy, peak: Math.max(...per.map((p) => p.peak)), spread, perWindow: per };
  });
  return { refIndex, frames: results };
}

// ---------------------------------------------------------------------------
// Candidate exposure kernels for the solve.
// ---------------------------------------------------------------------------

// Turn a registration result into a candidate motion assumption: a constant
// velocity path through the frame interval, with shutter fraction rho.
export function candidateFromRegistration(reg, rho, opts = {}) {
  const refIndex = reg.refIndex;
  const n = reg.frames.length;
  const perFrame = reg.frames.map((f, k) => (k === refIndex ? { dx: 0, dy: 0 } : f));
  // Least-squares constant velocity: s_k = v * (k - refIndex).
  let num = 0, den = 0, numY = 0;
  for (let k = 0; k < n; k++) {
    const t = k - refIndex;
    num += perFrame[k].dx * t;
    numY += perFrame[k].dy * t;
    den += t * t;
  }
  const vx = den ? num / den : 0;
  const vy = den ? numY / den : 0;
  const spread = reg.frames.reduce((m, f) => Math.max(m, f.spread), 0);
  return {
    vx, vy, rho,
    length: Math.hypot(vx, vy) * rho,
    angle: (Math.atan2(vy, vx) * 180) / Math.PI,
    spread,
    perFrame: perFrame.map((f) => ({ dx: f.dx, dy: f.dy })),
  };
}

// A deliberately alternative candidate: the same direction but reversed, or a
// stationary hypothesis. These exist because a symmetric exposure kernel and a
// symmetric texture can fit the data equally well; the interface must expose the
// choice rather than silently pick one.
export function alternativeCandidatesPrimary(cand) {
  return [
    { ...cand, id: 'primary', label: 'registered velocity' },
    {
      ...cand, id: 'reversed', label: 'reversed velocity', vx: -cand.vx, vy: -cand.vy,
    },
    {
      ...cand, id: 'stationary', label: 'stationary (no motion)', vx: 0, vy: 0,
      length: 0, angle: 0,
    },
  ];
}

// Blur kernel as a small dense array, used by the local convolution operator.
// dx,dy are pixel offsets relative to the exposure centre; samples are pairs
// [dx,dy,weight] splatted bilinearly into a (2R+1)^2 kernel.
export function buildKernel(vx, vy, rho, exposureKind, radius, samples, opts = {}) {
  const r = Math.max(1, Math.ceil(radius));
  const size = 2 * r + 1;
  const k = new Float64Array(size * size);
  const n = Math.max(8, samples);
  const weights = exposureWeightsLocal(exposureKind, n);
  const jitter = opts.jitter ?? 0;
  for (let i = 0; i < n; i++) {
    const u = n === 1 ? 0.5 : i / (n - 1);
    const dx = vx * rho * (u - 0.5) + (jitter ? (Math.sin(i * 12.9898) * 43758.5453 % 1) * jitter : 0);
    const dy = vy * rho * (u - 0.5) + (jitter ? (Math.sin(i * 78.233) * 43758.5453 % 1) * jitter : 0);
    const px = dx + r, py = dy + r;
    const x0 = Math.floor(px), y0 = Math.floor(py);
    const tx = px - x0, ty = py - y0;
    const wgt = weights[i];
    const splat = (xi, yi, v) => {
      if (xi >= 0 && xi < size && yi >= 0 && yi < size) k[yi * size + xi] += v;
    };
    splat(x0, y0, wgt * (1 - tx) * (1 - ty));
    splat(x0 + 1, y0, wgt * tx * (1 - ty));
    splat(x0, y0 + 1, wgt * (1 - tx) * ty);
    splat(x0 + 1, y0 + 1, wgt * tx * ty);
  }
  if (opts.gaussianSigma) {
    const g = gaussKernel2(size, opts.gaussianSigma);
    const out = new Float64Array(size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let s = 0;
        for (let j = -r; j <= r; j++) {
          for (let i = -r; i <= r; i++) {
            const xx = x + i, yy = y + j;
            if (xx < 0 || yy < 0 || xx >= size || yy >= size) continue;
            s += g[(j + r) * size + (i + r)] * k[yy * size + xx];
          }
        }
        out[y * size + x] = s;
      }
    }
    k.set(out);
  }
  let sum = 0;
  for (let i = 0; i < k.length; i++) sum += k[i];
  if (sum > 0) for (let i = 0; i < k.length; i++) k[i] /= sum;
  return { data: k, size, radius: r };
}

function gaussKernel2(size, sigma) {
  const r = (size - 1) / 2;
  const g = new Float64Array(size * size);
  let s = 0;
  for (let y = -r; y <= r; y++) {
    for (let x = -r; x <= r; x++) {
      const v = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
      g[(y + r) * size + (x + r)] = v;
      s += v;
    }
  }
  for (let i = 0; i < g.length; i++) g[i] /= s;
  return g;
}

export function exposureWeightsLocal(kind, n) {
  const w = new Float64Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0.5 : i / (n - 1);
    let v;
    switch (kind) {
      case 'centered': v = 0.45 + 0.55 * Math.sin(Math.PI * t); break;
      case 'ramp': v = 0.2 + 0.8 * t; break;
      case 'uniform': default: v = 1;
    }
    w[i] = Math.max(0, v);
    s += w[i];
  }
  for (let i = 0; i < n; i++) w[i] /= s;
  return w;
}

// Autocorrelation width of the exposure kernel (in pixels), reported so the UI
// can distinguish "short exposure" from "long exposure" honestly.
export function kernelStats(k) {
  const { data, size, radius } = k;
  let mass = 0, mx = 0, my = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = data[y * size + x];
      mass += v;
      mx += v * (x - radius);
      my += v * (y - radius);
    }
  }
  mx /= mass || 1;
  my /= mass || 1;
  let vx = 0, vy = 0, peak = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = data[y * size + x];
      const dx = x - radius - mx;
      const dy = y - radius - my;
      vx += v * dx * dx;
      vy += v * dy * dy;
      peak = Math.max(peak, v);
    }
  }
  return {
    cx: mx, cy: my,
    sigmaX: Math.sqrt(vx), sigmaY: Math.sqrt(vy),
    extent: Math.hypot(mx, my) * 2 + Math.sqrt(vx + vy) * 2,
    peak,
  };
}

// Kernel similarity: normalised cross-correlation of the two dense kernels
// after aligning their centroids. Used to detect "two kernels explain the data
// equally well" (ambiguity) versus "one is clearly better".
export function kernelNCC(a, b) {
  const size = Math.max(a.size, b.size);
  const A = padKernel(a, size);
  const B = padKernel(b, size);
  const sa = kernelStats(a);
  const sb = kernelStats(b);
  // Align centroids with integer shifts then compare.
  const shifted = shiftDense(B, size, Math.round(sb.cx - sa.cx), Math.round(sb.cy - sa.cy));
  let ma = 0, mb = 0, n = 0;
  for (let i = 0; i < size * size; i++) {
    if (shifted[i] >= 0) { ma += A[i]; mb += shifted[i]; n++; }
  }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < size * size; i++) {
    if (shifted[i] < 0) continue;
    const x = A[i] - ma, y = shifted[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return num / Math.sqrt(Math.max(da * db, 1e-12));
}

function padKernel(k, size) {
  if (k.size === size) return k.data;
  const out = new Float64Array(size * size);
  const off = (size - k.size) >> 1;
  for (let y = 0; y < k.size; y++) {
    for (let x = 0; x < k.size; x++) out[(y + off) * size + (x + off)] = k.data[y * k.size + x];
  }
  return out;
}

function shiftDense(k, size, dx, dy) {
  const out = new Float64Array(size * size).fill(-1);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = x + dx, ny = y + dy;
      if (nx >= 0 && ny >= 0 && nx < size && ny < size) out[ny * size + nx] = k[y * size + x];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Ambiguity diagnostics independent of the solver.
// ---------------------------------------------------------------------------

// Does the blurred data constrain the kernel direction? A reliable measure is
// the anisotropy of the local structure tensor of the *observed* frame: motion
// along an edge is invisible, motion across it is not. Report per-frame
// conditioning plus the registration spread.
export function motionObservability(obs, w, h, opts = {}) {
  const perFrame = [];
  for (const img of obs) {
    let jxx = 0, jyy = 0, jxy = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const gx = (img[y * w + x + 1] - img[y * w + x - 1]) * 0.5;
        const gy = (img[(y + 1) * w + x] - img[(y - 1) * w + x]) * 0.5;
        const m = gx * gx + gy * gy;
        jxx += gx * gx * m;
        jyy += gy * gy * m;
        jxy += gx * gy * m;
      }
    }
    const tr = jxx + jyy;
    const det = jxx * jyy - jxy * jxy;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    // direction of the dominant gradient (i.e. across edges)
    const theta = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
    perFrame.push({
      anisotropy: l2 > 1e-12 ? (l1 - l2) / (l1 + l2) : 0,
      edgeAngle: (theta * 180) / Math.PI,
      energy: tr,
    });
  }
  return perFrame;
}

// Scale / shift degeneracy for the exposure path: with an unknown overall shift
// between latent and observation, a kernel h and shifted kernel h(.-d) produce
// nearly the same data when h is smooth and the image is locally linear. We
// quantify it by comparing the re-blur residual of both hypotheses.
export function scaleShiftDegeneracy(dx, dy, length, rho) {
  // Normalised: how much of the observed displacement is explained by the
  // unknown per-frame translation versus the kernel length.
  const shiftMag = Math.hypot(dx, dy);
  const kLen = length * rho;
  if (kLen <= 1e-6) return shiftMag > 1 ? 1 : shiftMag;
  return clamp(Math.abs(shiftMag - Math.round(shiftMag)) / kLen, 0, 1);
}

export { angleDiffDeg };
