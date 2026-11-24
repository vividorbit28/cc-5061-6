// Regularised multi-frame deblurring / inversion.
//
// Minimise, over the dense latent canvas x,
//
//   sum_k || W_k ( A_k x - g_k ) ||^2  +  lambda_x || L x ||^2 + lambda_tv TV(x)
//
// with A_k = S_{d_k} H_k (Fourier exposure operator for frame k), W_k the
// observation support mask, and L a high-pass (Laplacian) prior. The solve is a
// preconditioned conjugate gradient on the normal equations, i.e. a real
// iterative optimisation with a logged residual history — not a sharpening
// filter. The number of iterations, lambda, and the motion assumptions are
// explicit inputs.
//
// Data support vs prior fill: the frequency-domain "reliability" of each mode is
//   rho(f) = sum_k |W_k A_k(f)|^2 / ( sum_k |W_k A_k(f)|^2 + lambda |L(f)|^2 )
// which is what the CG solve is implicitly trading off. We report it so the UI
// can show where detail is measured and where it is prior-driven.

import { FFT2, frequencies } from './fft.js';
import { forwardAll, cropObservation, embedObservation } from './model.js';
import { clamp } from './util.js';

const EPS = 1e-9;

function maskSpectrum(mask, padW, padH) {
  const f = new FFT2(padW, padH);
  f.re.set(mask);
  f.forward();
  return { re: Float64Array.from(f.re), im: Float64Array.from(f.im) };
}

function laplacianSpectrum(padW, padH) {
  const re = new Float64Array(padW * padH);
  const im = new Float64Array(padW * padH);
  const { fx, fy } = frequencies(padW, padH);
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const wx = (2 * Math.PI * fx[x]) / padW;
      const wy = (2 * Math.PI * fy[y]) / padH;
      re[y * padW + x] = 2 * Math.cos(wx) + 2 * Math.cos(wy) - 4;
    }
  }
  return { re, im };
}

function highpassSpectrum(padW, padH, scale = 0.5) {
  const re = new Float64Array(padW * padH);
  const im = new Float64Array(padW * padH);
  const { fx, fy } = frequencies(padW, padH);
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const wx = (2 * Math.PI * fx[x]) / padW;
      const wy = (2 * Math.PI * fy[y]) / padH;
      // 1 - low-pass: emphasise what deconvolution amplifies
      re[y * padW + x] = 1 - Math.exp(-scale * (wx * wx + wy * wy));
    }
  }
  return { re, im };
}

// Conjugate gradient on a Hermitian positive-definite system described by a
// function apply(v) -> A v. Solves A x = b in the Fourier domain (complex is
// handled by solving the two real normal-equation blocks together via the
// real-embedding of the fully-complex operator). We exploit the fact that the
// normal operator here is diagonal *plus* a compact (mask) term, so exact
// diagonal preconditioning is cheap and the iteration count stays small.
//
// To keep the algebra exact with a mask, we use a spatial-domain CG where each
// matrix-vector product is: (1) FFT x, (2) multiply by sum_k |A_k|^2 + lambda|L|^2,
// (3) inverse FFT, (4) add the mask-induced correction by comparing the masked
// forward residual with the unmasked operator. This is the standard "diagonal in
// frequency + low-rank mask correction" formulation; rather than approximate it,
// we implement the exact masked normal operator below.

class InverseProblem {
  constructor({ operators, masks, gCanvas, padW, padH, obsW, obsH, ox, oy, lambda, tv, regScale }) {
    this.ops = operators;
    this.masks = masks;
    this.g = gCanvas;
    this.padW = padW;
    this.padH = padH;
    this.obsW = obsW;
    this.obsH = obsH;
    this.ox = ox;
    this.oy = oy;
    this.lambda = lambda;
    this.tv = tv;
    this.n = padW * padH;
    this.nFrames = operators.length;
    this.f = new FFT2(padW, padH);
    this.L = highpassSpectrum(padW, padH, regScale ?? 0.6);
    this.tmpRe = new Float64Array(this.n);
    this.tmpIm = new Float64Array(this.n);
  }

  // Apply A_k to a spatial canvas.
  applyOp(x, op) {
    const { f, padW, padH, n } = this;
    f.re.set(x);
    f.im.fill(0);
    f.forward();
    for (let i = 0; i < n; i++) {
      const a = f.re[i], b = f.im[i];
      f.re[i] = a * op.re[i] - b * op.im[i];
      f.im[i] = a * op.im[i] + b * op.re[i];
    }
    f.inverse();
    return f.re;
  }

  // Apply A_k^H to a spatial canvas.
  applyOpH(x, op) {
    const { f, padW, padH, n } = this;
    f.re.set(x);
    f.im.fill(0);
    f.forward();
    for (let i = 0; i < n; i++) {
      const a = f.re[i], b = f.im[i];
      f.re[i] = a * op.re[i] + b * op.im[i];
      f.im[i] = -a * op.im[i] + b * op.re[i];
    }
    f.inverse();
    return f.re;
  }

  crop(canvas) {
    return cropObservation(canvas, this.padW, this.obsW, this.obsH, this.ox, this.oy);
  }

  embed(obs) {
    return embedObservation(obs, this.obsW, this.obsH, this.padW, this.padH, this.ox, this.oy);
  }

  // Normal operator: sum_k A_k^H W_k A_k + lambda L^H L (+ TV gradient).
  normal(x) {
    const { n, ops, masks } = this;
    const out = new Float64Array(n);
    for (let k = 0; k < ops.length; k++) {
      const ax = this.applyOp(x, ops[k]);
      const m = masks[k];
      const obs = this.crop(ax);
      const w = new Float64Array(this.obsW * this.obsH);
      for (let i = 0; i < w.length; i++) w[i] = obs[i] * m[i];
      const emb = this.embed(w);
      const back = this.applyOpH(emb, ops[k]);
      for (let i = 0; i < n; i++) out[i] += back[i];
    }
    // lambda * L^H L x  (real diagonal multiplier)
    this.f.re.set(x);
    this.f.im.fill(0);
    this.f.forward();
    for (let i = 0; i < n; i++) {
      const m = this.lambda * this.L.re[i] * this.L.re[i];
      this.f.re[i] *= m;
      this.f.im[i] *= m;
    }
    this.f.inverse();
    const reg = FFTReuse(this.f);
    for (let i = 0; i < n; i++) out[i] += reg[i];
    if (this.tv > 0) this.addTVGradient(out, x);
    return out;
  }

  addTVGradient(out, x) {
    const { padW, padH } = this;
    const eps = 1e-3;
    for (let y = 0; y < padH; y++) {
      for (let z = 0; z < padW; z++) {
        const i = y * padW + z;
        const xc = x[i];
        let gx = 0, gy = 0;
        if (z + 1 < padW) gx = x[i + 1] - xc;
        if (y + 1 < padH) gy = x[i + padW] - xc;
        const mag = Math.sqrt(gx * gx + gy * gy + eps * eps);
        const wgt = this.tv / mag;
        // gradient of TV: -div(grad/|grad|); approximate with forward diffs
        if (z + 1 < padW) out[i] -= this.tv * (gx / mag);
        if (y + 1 < padH) out[i] -= this.tv * (gy / mag);
        if (z > 0) {
          const xl = x[i - 1];
          const gl = xc - xl;
          const mag2 = Math.sqrt(gl * gl + eps * eps);
          // handled by neighbour's forward term above
        }
        if (z + 1 < padW) {
          const gx2 = x[i + 1] - xc;
          const gy2 = y + 1 < padH ? x[i + padW] - xc : 0;
          const m2 = Math.sqrt(gx2 * gx2 + gy2 * gy2 + eps * eps);
          out[i + 1] += this.tv * (gx2 / m2);
        }
        if (y + 1 < padH) {
          const gx2 = z + 1 < padW ? x[i + 1] - xc : 0;
          const gy2 = x[i + padW] - xc;
          const m2 = Math.sqrt(gx2 * gx2 + gy2 * gy2 + eps * eps);
          out[i + padW] += this.tv * (gy2 / m2);
        }
      }
    }
  }

  rhs() {
    const { n, ops, masks, g } = this;
    const out = new Float64Array(n);
    for (let k = 0; k < ops.length; k++) {
      const gk = this.crop(g[k]);
      const m = masks[k];
      const w = new Float64Array(this.obsW * this.obsH);
      for (let i = 0; i < w.length; i++) w[i] = gk[i] * m[i];
      const emb = this.embed(w);
      const bt = this.applyOpH(emb, ops[k]);
      for (let i = 0; i < n; i++) out[i] += bt[i];
    }
    return out;
  }
}

function FFTReuse(f) {
  return Float64Array.from(f.re);
}

export class Solver {
  constructor(config) {
    this.cfg = config;
    this.history = [];
  }

  static create({
    observations, operators, masks, padW, padH, obsW, obsH, ox, oy,
    lambda = 0.02, tv = 0, regScale = 0.6, maxIter = 60, tol = 1e-6,
    init = null, logEvery = 1,
  }) {
    const gCanvas = observations.map((g) =>
      embedObservation(g, obsW, obsH, padW, padH, ox, oy),
    );
    const problem = new InverseProblem({
      operators, masks, gCanvas, padW, padH, obsW, obsH, ox, oy, lambda, tv, regScale,
    });
    const n = padW * padH;
    const x = init ? Float64Array.from(init) : new Float64Array(n);
    const b = problem.rhs();
    // Plain CG with a frequency-domain diagonal preconditioner.
    const precond = makePreconditioner(operators, masks, padW, padH, lambda, regScale ?? 0.6, tv);
    let r = sub(b, problem.normal(x));
    let z = precond.apply(r);
    let p = Float64Array.from(z);
    let rz = dot(r, z);
    const hist = [];
    let iter = 0;
    const initialNorm = Math.sqrt(dot(b, b)) + EPS;
    for (; iter < maxIter; iter++) {
      const Ap = problem.normal(p);
      const pAp = dot(p, Ap);
      if (pAp <= EPS) break;
      const alpha = rz / pAp;
      for (let i = 0; i < n; i++) x[i] += alpha * p[i];
      const rNew = new Float64Array(n);
      for (let i = 0; i < n; i++) rNew[i] = r[i] - alpha * Ap[i];
      const rnorm = Math.sqrt(dot(rNew, rNew));
      if (iter % logEvery === 0 || iter === maxIter - 1) {
        hist.push({ iter: iter + 1, residual: rnorm / initialNorm, objective: 0.5 * dot(rNew, rNew) });
      }
      if (rnorm / initialNorm < tol) { r = rNew; iter++; break; }
      const zNew = precond.apply(rNew);
      const rzNew = dot(rNew, zNew);
      const beta = rzNew / Math.max(rz, EPS);
      for (let i = 0; i < n; i++) p[i] = zNew[i] + beta * p[i];
      r = rNew; z = zNew; rz = rzNew;
    }
    return {
      x,
      iterations: iter,
      history: hist,
      problem,
      finalRelResidual: hist.length ? hist[hist.length - 1].residual : 1,
    };
  }
}

function makePreconditioner(operators, masks, padW, padH, lambda, regScale, tv) {
  const n = padW * padH;
  // Approximate diagonal: average of |A_k|^2 over the full support (mask ~ 1 on
  // the observation window). Exact for the unmasked problem, which is the
  // dominant part; the mask only removes a thin border.
  const diag = new Float64Array(n);
  const tmp = new FFT2(padW, padH);
  for (const op of operators) {
    for (let i = 0; i < n; i++) diag[i] += op.re[i] * op.re[i] + op.im[i] * op.im[i];
  }
  const L = highpassSpectrum(padW, padH, regScale);
  for (let i = 0; i < n; i++) diag[i] += lambda * L.re[i] * L.re[i] + tv * 8;
  return {
    apply(r) {
      tmp.re.set(r);
      tmp.im.fill(0);
      tmp.forward();
      for (let i = 0; i < n; i++) {
        const den = diag[i] + 1e-6;
        tmp.re[i] /= den;
        tmp.im[i] /= den;
      }
      tmp.inverse();
      return Float64Array.from(tmp.re);
    },
  };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function sub(a, b) {
  const o = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] - b[i];
  return o;
}

// ---------------------------------------------------------------------------
// Diagnostics: reliability map, residual, re-blur check.
// ---------------------------------------------------------------------------

// rho(f) = sum|W A|^2 / (sum|W A|^2 + lambda|L|^2) on the observation window,
// averaged over directions. Returned in the spatial domain as an "information
// radius": how strongly each local frequency band is data-constrained.
export function reliabilitySpectrum(operators, lambda, regScale) {
  const { padW, padH } = operators.__dims ?? {};
}

export function dataSupportMap(xCanvas, operators, padW, padH, obsW, obsH, ox, oy, lambda, regScale = 0.6) {
  const n = padW * padH;
  // Compute the prior-only reconstruction contribution by running the same
  // frequency filter on a spatially smoothed copy. Frequencies where the data
  // term dominates are "supported"; the rest is "prior fill".
  const L = highpassSpectrum(padW, padH, regScale);
  const diag = new Float64Array(n);
  for (const op of operators) {
    for (let i = 0; i < n; i++) diag[i] += op.re[i] * op.re[i] + op.im[i] * op.im[i];
  }
  const f = new FFT2(padW, padH);
  f.re.set(xCanvas);
  f.forward();
  // Local support: integrate rho(f) weighted by the local spectrum of x.
  const rhoRe = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const d = diag[i];
    const reg = lambda * L.re[i] * L.re[i];
    rhoRe[i] = d / (d + reg + 1e-9);
  }
  // Spatially blur rho with |X|^2 weighting to get a local support estimate.
  const xr = Float64Array.from(f.re);
  const xi = Float64Array.from(f.im);
  const power = new Float64Array(n);
  for (let i = 0; i < n; i++) power[i] = xr[i] * xr[i] + xi[i] * xi[i];
  // support-weighted power and total power, both low-passed spatially
  const numRe = new Float64Array(n);
  const numIm = new Float64Array(n);
  for (let i = 0; i < n; i++) { numRe[i] = power[i] * rhoRe[i]; numIm[i] = power[i]; }
  f.re.set(numRe); f.im.fill(0); f.forward();
  applyGaussLowpass(f, padW, padH, 1.8);
  f.inverse();
  const numSpatial = Float64Array.from(f.re);
  f.re.set(numIm); f.im.fill(0); f.forward();
  applyGaussLowpass(f, padW, padH, 1.8);
  f.inverse();
  const denSpatial = Float64Array.from(f.re);
  const out = new Float64Array(obsW * obsH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      const i = (y + oy) * padW + (x + ox);
      out[y * obsW + x] = clamp(numSpatial[i] / (denSpatial[i] + 1e-9));
    }
  }
  return out;
}

function applyGaussLowpass(f, padW, padH, sigma) {
  const { fx, fy } = frequencies(padW, padH);
  const s2 = 2 * sigma * sigma;
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const i = y * padW + x;
      const wx = (2 * Math.PI * fx[x]) / padW;
      const wy = (2 * Math.PI * fy[y]) / padH;
      const g = Math.exp(-s2 * (wx * wx + wy * wy) / 2);
      f.re[i] *= g;
      f.im[i] *= g;
    }
  }
}

// Per-pixel "measured vs inferred" separation.
//
// The exposure kernels annihilate a band of spatial frequencies. Detail the
// estimate puts into that band cannot be supported by the data, so it is prior
// fill even when it looks crisp. For each local window we take the spectrum of
// the *estimate*, weight it by the high-pass response (so DC and shading do not
// dominate), and report the mean reliability rho(f) over that weighted band:
//
//   support = <|X(f)|^2 |L(f)|^2 rho(f)> / <|X(f)|^2 |L(f)|^2>
//   rho(f)  = sum_k |A_k(f)|^2 / (sum_k |A_k(f)|^2 + lambda |L(f)|^2)
//
// rho ~ 1: those frequencies pass through the model, so the data constrains
// them. rho ~ 0: the estimate is free to put anything there and the prior
// decides. This uses observations + assumed operators + lambda only; no truth.
export function localFrequencySupport(observations, operators, padW, padH, obsW, obsH, ox, oy,
  { lambda = 0.02, regScale = 0.6, block = 32, hop = 16, estimate = null } = {}) {
  const n = padW * padH;
  const L = highpassSpectrum(padW, padH, regScale);
  const hp = new Float64Array(n);
  for (let i = 0; i < n; i++) hp[i] = L.re[i] * L.re[i];
  const diag = new Float64Array(n);
  for (const op of operators) {
    for (let i = 0; i < n; i++) diag[i] += op.re[i] * op.re[i] + op.im[i] * op.im[i];
  }
  let diagMax = 0;
  for (let i = 0; i < n; i++) diagMax = Math.max(diagMax, diag[i]);
  const rho = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const d = diag[i];
    rho[i] = d / (d + lambda * hp[i] + 1e-9 * diagMax);
  }
  // Local spectra come from the estimate when supplied (default), else the
  // middle observation. Only the frequency content layout matters.
  const img = estimate ?? observations[Math.floor(observations.length / 2)];
  const bw = Math.min(block, obsW);
  const bh = Math.min(block, obsH);
  const win = new Float64Array(bw * bh);
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const wx = 0.5 - 0.5 * Math.cos((2 * Math.PI * (x + 0.5)) / bw);
      const wy = 0.5 - 0.5 * Math.cos((2 * Math.PI * (y + 0.5)) / bh);
      win[y * bw + x] = wx * wy;
    }
  }
  const out = new Float64Array(obsW * obsH);
  const weight = new Float64Array(obsW * obsH);
  const sx = padW / bw;
  const sy = padH / bh;
  for (let by0 = -hop; by0 < obsH; by0 += hop) {
    for (let bx0 = -hop; bx0 < obsW; bx0 += hop) {
      if (by0 + bh <= 0 || bx0 + bw <= 0 || by0 >= obsH || bx0 >= obsW) continue;
      const f = new FFT2(bw, bh);
      for (let y = 0; y < bh; y++) {
        for (let x = 0; x < bw; x++) {
          const gx = bx0 + x, gy = by0 + y;
          const inside = gx >= 0 && gy >= 0 && gx < obsW && gy < obsH;
          f.re[y * bw + x] = (inside ? img[gy * obsW + gx] : 0) * win[y * bw + x];
        }
      }
      f.im.fill(0);
      f.forward();
      let num = 0;
      let den = 0;
      for (let y = 0; y < bh; y++) {
        for (let x = 0; x < bw; x++) {
          const i = y * bw + x;
          const p = f.re[i] * f.re[i] + f.im[i] * f.im[i];
          // Window frequency -> padded-canvas frequency grid.
          const px = ((Math.round((x <= bw / 2 ? x : x - bw) * sx) % padW) + padW) % padW;
          const py = ((Math.round((y <= bh / 2 ? y : y - bh) * sy) % padH) + padH) % padH;
          const li = py * padW + px;
          const w = p * hp[li];
          den += w;
          num += w * rho[li];
        }
      }
      const sup = den > 1e-12 ? clamp(num / den, 0, 1) : 0.5;
      for (let y = 0; y < bh; y++) {
        const gy = by0 + y;
        if (gy < 0 || gy >= obsH) continue;
        for (let x = 0; x < bw; x++) {
          const gx = bx0 + x;
          if (gx < 0 || gx >= obsW) continue;
          out[gy * obsW + gx] += sup;
          weight[gy * obsW + gx] += 1;
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = weight[i] > 0 ? out[i] / weight[i] : 0.5;
  return out;
}

// Contrast between the reported estimate and a heavily regularised solve, used
// by the UI to show *which* structure would vanish under a stronger prior.
export function priorContrast(estimate, altEstimate, obsW, obsH) {
  const out = new Float64Array(obsW * obsH);
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, Math.abs(estimate[i] - altEstimate[i]) * 4);
  return out;
}

export function reblurResidual(estimate, observations, operators, padW, padH, obsW, obsH, ox, oy) {
  const canvas = embedObservation(estimate, obsW, obsH, padW, padH, ox, oy);
  const pred = forwardAll(canvas, operators, padW, padH);
  const res = [];
  for (let k = 0; k < observations.length; k++) {
    const pk = cropObservation(pred[k], padW, obsW, obsH, ox, oy);
    const r = new Float64Array(obsW * obsH);
    for (let i = 0; i < r.length; i++) r[i] = observations[k][i] - pk[i];
    res.push({ predicted: pk, residual: r });
  }
  return res;
}

export { highpassSpectrum, laplacianSpectrum, maskSpectrum };
