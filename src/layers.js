// Layered exposure-blur inversion.
//
// The single-motion model in model.js assumes every observed pixel moves along
// one global path. Real frames contain independently moving layers (a moving
// foreground object over a background that moves with the camera). This module
// extends the inversion to a finite, capped number of motion layers.
//
// Each layer l has:
//   x_l : sharp latent on the shared padded canvas (reference coordinates)
//   h_{l,k} : its own exposure kernel, path-integrated over frame k's shutter
//   m_l(tau) : a soft coverage (alpha) region that follows the layer over time
//
// Frame k is an alpha composite of the layers over the shutter interval. With
// J exposure samples t_j = k + (u_j - 1/2) rho and shutter weights kappa_j:
//
//   z_k(p) = sum_j kappa_j * [
//       sum_l m_l(p - d_l(t_j)) * S_{d_l(t_j)} ( h_{l,k} * x_l ) (p)
//     ] / sum_j kappa_j sum_l m_l(p - d_l(t_j))
//   g_k = crop(z_k)
//
// The layer boundary is therefore *inside* the imaging model: a pixel covered
// by the foreground during part of the exposure and the background during the
// rest receives a time-weighted mix. We never blur the whole frame with several
// kernels and stitch the results afterwards.
//
// Two declared approximations are implemented:
//   - 'time-sampled' (exact for the discrete model): every exposure sample
//     re-evaluates the shifted coverage. Used by the clip generator and by the
//     forward-verify test, so generator and inversion close the loop.
//   - 'frozen-mask' (default for inversion): the coverage of frame k is frozen
//     at the frame centre while the layer image still passes through its full
//     exposure operator. This makes the normal operator a sum of
//     mask * S H terms (FFT-friendly CG). The boundary error this introduces is
//     measured in the tests as a boundary-vs-interior residual ratio.
//
// The number of layers and the motion degrees of freedom are capped; a layer
// that the observations cannot identify is reported with an uncertainty region
// instead of being fitted freely.

import { FFT2, frequencies } from './fft.js';
import { clamp, smoothstep } from './util.js';
import { frameKernel, operatorSpectrum, embedObservation, cropObservation } from './model.js';
import { makeScene, latentFromScene, GEOM, quantize8, addNoise, makeParametricPath } from './synth.js';

export const MAX_LAYERS = 3;
const EPS = 1e-9;

// ---------------------------------------------------------------------------
// Coverage masks
// ---------------------------------------------------------------------------

// Soft coverage mask on the padded canvas for a shape in *reference* coords.
// shape: { cx, cy, rx, ry, angleRad } (rx/ry half extents on the padded grid).
export function shapeMask(padW, padH, shape, soft = 1.2) {
  const m = new Float64Array(padW * padH);
  const c = Math.cos(-(shape.angleRad || 0));
  const s = Math.sin(-(shape.angleRad || 0));
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const dx = x + 0.5 - shape.cx;
      const dy = y + 0.5 - shape.cy;
      const lx = dx * c - dy * s;
      const ly = dx * s + dy * c;
      // signed-ish distance to the ellipse/rounded-rect edge
      const ex = Math.abs(lx) - shape.rx;
      const ey = Math.abs(ly) - shape.ry;
      const ax = Math.max(ex, 0), ay = Math.max(ey, 0);
      const d = Math.sqrt(ax * ax + ay * ay) + Math.min(Math.max(ex, ey), 0);
      m[y * padW + x] = 1 - smoothstep(clamp((d + soft) / (2 * soft), 0, 1));
    }
  }
  return m;
}

// Shift a soft mask spatially by (dx, dy) using bilinear interpolation. The
// canvas is zero outside its bounds (a layer leaving the frame uncovers the
// layer behind it). Positive dx moves coverage to the right.
export function shiftMask(mask, padW, padH, dx, dy) {
  const out = new Float64Array(padW * padH);
  const x0 = Math.floor(dx), y0 = Math.floor(dy);
  const tx = dx - x0, ty = dy - y0;
  const at = (x, y) => (x < 0 || y < 0 || x >= padW || y >= padH ? 0 : mask[y * padW + x]);
  for (let y = 0; y < padH; y++) {
    const sy = y - y0;
    for (let x = 0; x < padW; x++) {
      const sx = x - x0;
      const v00 = at(sx, sy), v10 = at(sx + 1, sy);
      const v01 = at(sx, sy + 1), v11 = at(sx + 1, sy + 1);
      out[y * padW + x] =
        v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) +
        v01 * (1 - tx) * ty + v11 * tx * ty;
    }
  }
  return out;
}

// Observation-grid coverage of a layer at frame centre, embedded onto the
// padded canvas (zero outside the observation window).
function frozenLayerMask(refMask, padW, padH, obsW, obsH, ox, oy, dxC, dyC) {
  const shifted = shiftMask(refMask, padW, padH, dxC, dyC);
  const out = new Float64Array(padW * padH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) out[(y + oy) * padW + (x + ox)] = shifted[(y + oy) * padW + (x + ox)];
  }
  return out;
}

// Normalise per-frame masks so the visible layers partition coverage: interior
// pixels get sum 1; outside the observation window everything stays zero.
function normaliseMasks(masks, padW, padH, obsW, obsH, ox, oy) {
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      const i = (y + oy) * padW + (x + ox);
      let sum = 0;
      for (const m of masks) sum += m[i];
      if (sum > EPS) for (const m of masks) m[i] /= sum;
    }
  }
}

// ---------------------------------------------------------------------------
// Layer paths and per-frame exposure operators
// ---------------------------------------------------------------------------

function layerPath(layer) {
  return makeParametricPath({
    kind: layer.kind ?? 'linear',
    vx: layer.vx ?? 0, vy: layer.vy ?? 0,
    amp: layer.amp ?? 0, freq: layer.freq ?? 0,
    omega: layer.omega ?? 0, pivotX: layer.pivotX ?? 0, pivotY: layer.pivotY ?? 0,
  });
}

// Build, for every frame, each layer's exposure operator and its coverage.
// 'frozen' freezes coverage at the frame centre; 'time' returns per-exposure-
// sample coverage for the exact composite forward.
export function buildLayeredOperators(layers, frames, refIndex, rho, exposureKind, geom = GEOM, mode = 'frozen') {
  const { padW, padH, obsW, obsH, ox, oy } = geom;
  const paths = layers.map(layerPath);
  const perFrame = [];
  for (let k = 0; k < frames; k++) {
    const frame = { index: k, layers: [] };
    for (let l = 0; l < layers.length; l++) {
      const path = paths[l];
      const speed = Math.hypot(layers[l].vx ?? 0, layers[l].vy ?? 0);
      const radius = Math.max(5, 9 + speed * rho);
      const kernel = frameKernel(path, k, rho, layers[l].exposureKind ?? exposureKind, radius, 32);
      const pRef = path(refIndex);
      const pK = path(k);
      const dxC = pK.x - pRef.x;
      const dyC = pK.y - pRef.y;
      const op = operatorSpectrum(kernel, dxC, dyC, padW, padH);
      const entry = { kernel, op, dxC, dyC };
      if (mode === 'frozen') {
        // A full-window layer (the background) keeps its coverage pinned to the
        // observation window: image motion is already encoded in the operator's
        // Fourier shift, and shifting the all-ones mask would erase the border
        // pixels that move out and manufacture a spurious partial mask. Only a
        // local foreground lock region (fixedCoverage !== true) translates with
        // the layer so its protected footprint follows the object over time.
        entry.maskCanvas = layers[l].fixedCoverage
          ? Float64Array.from(layers[l].maskCanvas)
          : frozenLayerMask(layers[l].maskCanvas, padW, padH, obsW, obsH, ox, oy, dxC, dyC);
      } else {
        entry.path = path;
        entry.pRef = pRef;
      }
      frame.layers.push(entry);
    }
    // Coverage is *not* renormalised: soft alpha edges (0 < m < 1) are the
    // declared boundary approximation and must keep their weights. The
    // background carries mask 1, so interior pixels still partition coverage.
    perFrame.push(frame);
  }
  return { paths, perFrame };
}

// ---------------------------------------------------------------------------
// Exact (time-sampled) composite forward used by the generator and verification
// ---------------------------------------------------------------------------

function applyShiftBlur(xCanvas, path, k, rho, exposureKind, padW, padH, samples) {
  const kernel = frameKernel(path, k, rho, exposureKind, Math.max(8, 14), samples);
  const op = operatorSpectrum(kernel, 0, 0, padW, padH);
  const f = new FFT2(padW, padH);
  f.re.set(xCanvas);
  f.forward();
  for (let i = 0; i < padW * padH; i++) {
    const a = f.re[i], b = f.im[i];
    f.re[i] = a * op.re[i] - b * op.im[i];
    f.im[i] = a * op.im[i] + b * op.re[i];
  }
  f.inverse();
  return f.re;
}

// Spatial bilinear translation (zero outside). Positive dx moves content right,
// matching the Fourier-shift convention. Used by the time-sampled generator so
// it does not need one FFT per exposure sample per layer.
function shiftImageSpatial(src, padW, padH, dx, dy) {
  const out = new Float64Array(padW * padH);
  const x0 = Math.floor(dx), y0 = Math.floor(dy);
  const tx = dx - x0, ty = dy - y0;
  const at = (x, y) => (x < 0 || y < 0 || x >= padW || y >= padH ? 0 : src[y * padW + x]);
  for (let y = 0; y < padH; y++) {
    const sy = y - y0;
    for (let x = 0; x < padW; x++) {
      const sx = x - x0;
      out[y * padW + x] =
        at(sx, sy) * (1 - tx) * (1 - ty) + at(sx + 1, sy) * tx * (1 - ty) +
        at(sx, sy + 1) * (1 - tx) * ty + at(sx + 1, sy + 1) * tx * ty;
    }
  }
  return out;
}

// Exact layered forward for one frame. Each exposure sample shifts every layer
// image to its sample-time position and composites with the sample-time
// coverage; the normalisation makes this a proper alpha blend.
function exactFrameForward(layerCanvases, layers, paths, k, refIndex, rho, exposureKind, padW, padH, samples) {
  const n = padW * padH;
  const acc = new Float64Array(n);
  const wsum = new Float64Array(n);
  // Blurred layer images (shutter-integrated, not yet translated).
  const blurred = layerCanvases.map((canvas, l) =>
    applyShiftBlur(canvas, paths[l], k, rho, layers[l].exposureKind ?? exposureKind, padW, padH, samples));
  for (let j = 0; j < samples; j++) {
    const u = samples === 1 ? 0.5 : j / (samples - 1);
    const tau = k + (u - 0.5) * rho;
    const w = 1 / samples;
    for (let l = 0; l < layers.length; l++) {
      const p = paths[l](tau);
      const pr = paths[l](refIndex);
      const dx = p.x - pr.x, dy = p.y - pr.y;
      const moved = shiftImageSpatial(blurred[l], padW, padH, dx, dy);
      const cover = shiftMask(layers[l].maskCanvas, padW, padH, dx, dy);
      for (let i = 0; i < n; i++) {
        acc[i] += w * cover[i] * moved[i];
        wsum[i] += w * cover[i];
      }
    }
  }
  for (let i = 0; i < n; i++) acc[i] = wsum[i] > EPS ? acc[i] / wsum[i] : 0;
  return acc;
}

// ---------------------------------------------------------------------------
// Frozen-mask forward for the inversion and for re-blur verification
// ---------------------------------------------------------------------------

export function frozenForward(xCanvases, built, geom = GEOM) {
  const { padW, padH, obsW, obsH, ox, oy } = geom;
  const f = new FFT2(padW, padH);
  const frames = [];
  for (const frame of built.perFrame) {
    const composite = new Float64Array(padW * padH);
    for (let l = 0; l < xCanvases.length; l++) {
      const entry = frame.layers[l];
      f.re.set(xCanvases[l]);
      f.im.fill(0);
      f.forward();
      for (let i = 0; i < padW * padH; i++) {
        const a = f.re[i], b = f.im[i];
        f.re[i] = a * entry.op.re[i] - b * entry.op.im[i];
        f.im[i] = a * entry.op.im[i] + b * entry.op.re[i];
      }
      f.inverse();
      const moved = f.re;
      const m = entry.maskCanvas;
      for (let i = 0; i < composite.length; i++) composite[i] += m[i] * moved[i];
    }
    frames.push(cropObservation(composite, padW, obsW, obsH, ox, oy));
  }
  return frames;
}

// ---------------------------------------------------------------------------
// Joint multi-layer CG solve (frozen-mask model)
//
// Minimise over the stacked latent x = (x_0, ..., x_{L-1}):
//   sum_k || W_k ( sum_l M_{l,k} A_{l,k} x_l - g_k ) ||^2
//     + lambda sum_l || L x_l ||^2
// Each layer gets its own exposure operator A_{l,k} and its own time-moving
// lock region M_{l,k}. The system is jointly normal-equation CG with a
// per-layer frequency diagonal preconditioner.
// ---------------------------------------------------------------------------

function highpassSpectrum(padW, padH, scale = 0.6) {
  const re = new Float64Array(padW * padH);
  const { fx, fy } = frequencies(padW, padH);
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const wx = (2 * Math.PI * fx[x]) / padW;
      const wy = (2 * Math.PI * fy[y]) / padH;
      re[y * padW + x] = 1 - Math.exp(-scale * (wx * wx + wy * wy));
    }
  }
  return re;
}

export class LayeredProblem {
  constructor({ built, gCanvas, dataMasks, geom, lambda, regScale }) {
    this.built = built;
    this.g = gCanvas; // per frame, padded canvas
    this.dataMasks = dataMasks; // per frame, padded canvas
    this.geom = geom;
    this.lambda = lambda;
    this.Lre = highpassSpectrum(geom.padW, geom.padH, regScale);
    this.nLayers = built.perFrame[0].layers.length;
    this.nFrames = built.perFrame.length;
    this.n = geom.padW * geom.padH;
    // Pool of reusable transforms. acquire()/release() keep nested operator
    // applications on independent buffers (a single shared buffer would let an
    // inner transform overwrite an outer one) without reallocating each call.
    this.pool = [];
    for (let i = 0; i < 4; i++) this.pool.push(new FFT2(geom.padW, geom.padH));
    this.depth = 0;
  }

  acquireFft() {
    if (this.depth >= this.pool.length) this.pool.push(new FFT2(this.geom.padW, this.geom.padH));
    return this.pool[this.depth++];
  }
  releaseFft() { this.depth = Math.max(0, this.depth - 1); }

  applyOp(x, op) {
    const { n } = this;
    const f = this.acquireFft();
    f.re.set(x); f.im.fill(0); f.forward();
    for (let i = 0; i < n; i++) {
      const a = f.re[i], b = f.im[i];
      f.re[i] = a * op.re[i] - b * op.im[i];
      f.im[i] = a * op.im[i] + b * op.re[i];
    }
    f.inverse();
    const out = Float64Array.from(f.re);
    this.releaseFft();
    return out;
  }

  applyOpH(x, op) {
    const { n } = this;
    const f = this.acquireFft();
    f.re.set(x); f.im.fill(0); f.forward();
    for (let i = 0; i < n; i++) {
      const a = f.re[i], b = f.im[i];
      f.re[i] = a * op.re[i] + b * op.im[i];
      f.im[i] = -a * op.im[i] + b * op.re[i];
    }
    f.inverse();
    const out = Float64Array.from(f.re);
    this.releaseFft();
    return out;
  }

  // Pure linear normal operator L(x) = sum_k A^H M W M A x + lambda L^H L x.
  // It contains NO data term, so it is genuinely linear (L(ax+by)=aL(x)+bL(y))
  // and is the operator the CG recursion and preconditioner are built for.
  linearNormal(xStack) {
    const out = xStack.map(() => new Float64Array(this.n));
    for (let k = 0; k < this.nFrames; k++) {
      const frame = this.built.perFrame[k];
      const pred = new Float64Array(this.n);
      for (let l = 0; l < this.nLayers; l++) {
        const ax = this.applyOp(xStack[l], frame.layers[l].op);
        const m = frame.layers[l].maskCanvas;
        for (let i = 0; i < this.n; i++) pred[i] += m[i] * ax[i];
      }
      const w = this.dataMasks[k];
      const r = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) r[i] = pred[i] * w[i];
      for (let l = 0; l < this.nLayers; l++) {
        const m = frame.layers[l].maskCanvas;
        const mr = new Float64Array(this.n);
        for (let i = 0; i < this.n; i++) mr[i] = m[i] * r[i];
        const back = this.applyOpH(mr, frame.layers[l].op);
        for (let i = 0; i < this.n; i++) out[l][i] += back[i];
      }
    }
    // Prior block, lambda L^H L (real diagonal multiplier).
    const pf = new FFT2(this.geom.padW, this.geom.padH);
    for (let l = 0; l < this.nLayers; l++) {
      pf.re.set(xStack[l]); pf.im.fill(0); pf.forward();
      for (let i = 0; i < this.n; i++) {
        const m = this.lambda * this.Lre[i] * this.Lre[i];
        pf.re[i] *= m;
        pf.im[i] *= m;
      }
      pf.inverse();
      for (let i = 0; i < this.n; i++) out[l][i] += pf.re[i];
    }
    return out;
  }

  // Affine normal-equations evaluation N(x) = L(x) - b, equal to the negative
  // data residual in stacked form. Kept for diagnostics; the solver uses
  // linearNormal + rhs directly so the CG recursion stays exactly linear.
  normal(xStack) {
    const lin = this.linearNormal(xStack);
    const b = this.rhs();
    return lin.map((v, l) => {
      const o = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) o[i] = v[i] - b[l][i];
      return o;
    });
  }

  rhs() {
    const out = Array.from({ length: this.nLayers }, () => new Float64Array(this.n));
    for (let k = 0; k < this.nFrames; k++) {
      const frame = this.built.perFrame[k];
      const w = this.dataMasks[k];
      const wg = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) wg[i] = this.g[k][i] * w[i];
      for (let l = 0; l < this.nLayers; l++) {
        const m = frame.layers[l].maskCanvas;
        const mwg = new Float64Array(this.n);
        for (let i = 0; i < this.n; i++) mwg[i] = m[i] * wg[i];
        const back = this.applyOpH(mwg, frame.layers[l].op);
        for (let i = 0; i < this.n; i++) out[l][i] += back[i];
      }
    }
    return out;
  }

  preconditioner() {
    const { padW, padH } = this.geom;
    const n = this.n;
    // Pure frequency-diagonal block preconditioner, identical in spirit to the
    // proven single-motion solver (solve.js): each block diagonal is
    //   sum_k |A_{l,k}|^2 + lambda |L|^2
    // The coverage masks enter only through the exact normal() operator; they
    // are deliberately NOT folded into this diagonal, because a spatially
    // varying mask cannot be represented by a frequency multiplier and doing so
    // was found to destroy the preconditioner's accuracy. For a full-coverage
    // layer this diagonal is the exact inverse of the unmasked normal operator,
    // which is the dominant (data-supported) part of the system.
    const diags = Array.from({ length: this.nLayers }, () => new Float64Array(n));
    for (let k = 0; k < this.nFrames; k++) {
      const frame = this.built.perFrame[k];
      for (let l = 0; l < this.nLayers; l++) {
        const op = frame.layers[l].op;
        for (let i = 0; i < n; i++) diags[l][i] += op.re[i] * op.re[i] + op.im[i] * op.im[i];
      }
    }
    for (let l = 0; l < this.nLayers; l++) {
      for (let i = 0; i < n; i++) diags[l][i] += this.lambda * this.Lre[i] * this.Lre[i] + 1e-6;
    }
    const f = new FFT2(padW, padH);
    return {
      apply(stack) {
        return stack.map((r, l) => {
          f.re.set(r); f.im.fill(0); f.forward();
          for (let i = 0; i < n; i++) { f.re[i] /= diags[l][i]; f.im[i] /= diags[l][i]; }
          f.inverse();
          return Float64Array.from(f.re);
        });
      },
    };
  }
}

function stackDot(a, b) {
  let s = 0;
  for (let l = 0; l < a.length; l++) for (let i = 0; i < a[l].length; i++) s += a[l][i] * b[l][i];
  return s;
}

export function solveLayered({ observations, built, layers, geom = GEOM, lambda = 0.02, regScale = 0.6, maxIter = 40, tol = 2e-5, init = null }) {
  const { padW, padH, obsW, obsH, ox, oy } = geom;
  const n = padW * padH;
  const gCanvas = observations.map((g) => embedObservation(g, obsW, obsH, padW, padH, ox, oy));
  const dataMasks = gCanvas.map(() => {
    const m = new Float64Array(n);
    for (let y = 0; y < obsH; y++) for (let x = 0; x < obsW; x++) m[(y + oy) * padW + (x + ox)] = 1;
    return m;
  });
  const problem = new LayeredProblem({ built, gCanvas, dataMasks, geom, lambda, regScale });
  const x = init
    ? init.map((c) => Float64Array.from(c))
    : Array.from({ length: problem.nLayers }, () => new Float64Array(n));
  const b = problem.rhs();
  const pre = problem.preconditioner();
  const residualOf = (xx) => {
    const nx = problem.linearNormal(xx);
    return b.map((v, l) => {
      const o = new Float64Array(n);
      for (let i = 0; i < n; i++) o[i] = v[i] - nx[l][i];
      return o;
    });
  };
  const bNorm = Math.sqrt(stackDot(b, b)) + EPS;
  const history = [];
  // Standard preconditioned CG on the stacked normal equations, matching the
  // proven single-motion solver: the residual is updated recursively and the
  // frequency-diagonal preconditioner makes convergence fast. As a safety net
  // against roundoff eroding conjugacy on the wide-spectrum layered system, the
  // TRUE residual is recomputed every `restart` iterations and the conjugate
  // direction is restarted from the fresh preconditioned residual.
  const restart = 15;
  let r = residualOf(x);
  let z = pre.apply(r);
  let p = z.map((v) => Float64Array.from(v));
  let rz = stackDot(r, z);
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const Ap = problem.linearNormal(p);
    const pAp = stackDot(p, Ap);
    if (!(Number.isFinite(pAp) && pAp > EPS)) break;
    const alpha = rz / pAp;
    for (let l = 0; l < x.length; l++) for (let i = 0; i < n; i++) x[l][i] += alpha * p[l][i];

    const doRestart = (iter + 1) % restart === 0;
    if (doRestart) {
      r = residualOf(x);
    } else {
      r = r.map((v, l) => {
        const o = new Float64Array(n);
        for (let i = 0; i < n; i++) o[i] = v[i] - alpha * Ap[l][i];
        return o;
      });
    }
    const rnorm = Math.sqrt(stackDot(r, r));
    history.push({ iter: iter + 1, residual: rnorm / bNorm });
    if (rnorm / bNorm < tol) { iter++; break; }

    z = pre.apply(r);
    if (doRestart) {
      p = z.map((v) => Float64Array.from(v));
      rz = stackDot(r, z);
    } else {
      const rzNew = stackDot(r, z);
      const beta = rzNew / Math.max(rz, EPS);
      p = z.map((v, l) => {
        const o = new Float64Array(n);
        for (let i = 0; i < n; i++) o[i] = v[i] + beta * p[l][i];
        return o;
      });
      rz = rzNew;
    }
  }
  return { xCanvases: x, iterations: iter, history, problem };
}

// ---------------------------------------------------------------------------
// Data-driven layer lock regions (no hand segmentation by default)
// ---------------------------------------------------------------------------

// Estimate a foreground lock region from the observations alone. The moving
// object creates local frame-to-frame differences that persist after the global
// (background) motion is compensated. `bgVel` is the estimated background
// velocity in obs px/frame. Returns a soft mask on the *observation* grid at the
// reference frame.
//
// Motion-compensated difference alone is contaminated by the background's own
// exposure blur (especially under a long shutter), so a raw median threshold
// over-segments. The candidate is therefore cleaned morphologically: an opening
// removes isolated blur specks, the largest connected blob is kept (the single
// foreground object), and a closing fills its interior. This stays purely
// data-driven — no hand-drawn mask is used by default.
export function estimateForegroundRegion(obs, bgVel, refIndex, w, h, opts = {}) {
  const strength = opts.strength ?? 1.0;
  const openR = opts.openRadius ?? 1;
  const closeR = opts.closeRadius ?? 3;
  const acc = new Float64Array(w * h);
  let count = 0;
  for (let k = 0; k < obs.length; k++) {
    if (k === refIndex) continue;
    const dt = k - refIndex;
    const dx = bgVel.x * dt;
    const dy = bgVel.y * dt;
    const warped = warpBilinear(obs[k], w, h, dx, dy);
    for (let i = 0; i < acc.length; i++) acc[i] += Math.abs(obs[refIndex][i] - warped[i]);
    count++;
  }
  if (!count) return acc;
  for (let i = 0; i < acc.length; i++) acc[i] /= count;

  // High-percentile cut: only the strongest, spatially coherent residual is a
  // foreground candidate. The soft score keeps alpha edges for the solver.
  const sorted = Float64Array.from(acc).sort();
  const p85 = sorted[Math.floor(sorted.length * 0.85)];
  const p97 = sorted[Math.floor(sorted.length * 0.97)] || p85 + 1e-3;
  const score = new Float64Array(w * h);
  for (let i = 0; i < acc.length; i++) score[i] = strength * smoothstep(clamp((acc[i] - p85) / Math.max(p97 - p85, 1e-6), 0, 1));

  // Hard candidate for morphology.
  let binary = new Uint8Array(w * h);
  for (let i = 0; i < acc.length; i++) binary[i] = acc[i] >= p85 ? 1 : 0;
  binary = morphOpen(binary, w, h, openR);
  // Drop blobs touching the observation border: those are warp fill/background
  // edge artifacts, not an interior moving object. Keep every remaining
  // interior blob rather than forcing the single largest, because under a long
  // shutter a genuinely present object can split into several coherent pieces.
  const kept = interiorComponents(binary, w, h);
  const cleaned = morphClose(kept, w, h, closeR);

  const out = new Float64Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = cleaned[i] ? score[i] : 0;
  // If cleanup removed everything (no coherent object), fall back to the soft
  // score rather than emitting an empty lock region.
  let mass = 0;
  for (const v of out) mass += v;
  if (mass < 1) return score;
  return out;
}

function dilate(bin, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let yy = -r; yy <= r && !v; yy++) for (let xx = -r; xx <= r && !v; xx++) {
      const nx = x + xx, ny = y + yy;
      if (nx >= 0 && ny >= 0 && nx < w && ny < h && bin[ny * w + nx]) v = 1;
    }
    out[y * w + x] = v;
  }
  return out;
}
function erode(bin, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 1;
    for (let yy = -r; yy <= r && v; yy++) for (let xx = -r; xx <= r && v; xx++) {
      const nx = x + xx, ny = y + yy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h || !bin[ny * w + nx]) v = 0;
    }
    out[y * w + x] = v;
  }
  return out;
}
function morphOpen(bin, w, h, r) { return r <= 0 ? bin : dilate(erode(bin, w, h, r), w, h, r); }
function morphClose(bin, w, h, r) { return r <= 0 ? bin : erode(dilate(bin, w, h, r), w, h, r); }

// 4-connected flood fill; keep only blobs that do not touch the image border
// (border-touching residuals are warp-fill artifacts). Returns a binary mask.
function interiorComponents(bin, w, h) {
  const labels = new Int32Array(w * h).fill(-1);
  const touchesBorder = [];
  const stack = [];
  for (let start = 0; start < bin.length; start++) {
    if (!bin[start] || labels[start] !== -1) continue;
    const id = touchesBorder.length;
    let border = false;
    labels[start] = id;
    stack.length = 0;
    stack.push(start);
    while (stack.length) {
      const i = stack.pop();
      const x = i % w, y = (i / w) | 0;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) border = true;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) {
        if (j >= 0 && bin[j] && labels[j] === -1) { labels[j] = id; stack.push(j); }
      }
    }
    touchesBorder.push(border);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < bin.length; i++) {
    const id = labels[i];
    if (id !== -1 && !touchesBorder[id]) out[i] = 1;
  }
  return out;
}

function warpBilinear(img, w, h, dx, dy) {
  const out = new Float64Array(w * h);
  const x0 = Math.floor(dx), y0 = Math.floor(dy);
  const tx = dx - x0, ty = dy - y0;
  const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : img[y * w + x]);
  for (let y = 0; y < h; y++) {
    const sy = y - y0;
    for (let x = 0; x < w; x++) {
      const sx = x - x0;
      out[y * w + x] =
        at(sx, sy) * (1 - tx) * (1 - ty) + at(sx + 1, sy) * tx * (1 - ty) +
        at(sx, sy + 1) * (1 - tx) * ty + at(sx + 1, sy + 1) * tx * ty;
    }
  }
  return out;
}

// Promote an observation-grid soft mask to a padded-canvas reference mask,
// applying a light spatial smooth so the lock region has soft boundaries.
function obsMaskToCanvas(obsMask, w, h, padW, padH, ox, oy, smoothPx = 1.5) {
  const c = new Float64Array(padW * padH);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) c[(y + oy) * padW + (x + ox)] = obsMask[y * w + x];
  if (smoothPx <= 0) return c;
  const r = Math.ceil(smoothPx);
  const out = new Float64Array(padW * padH);
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      let s = 0, n = 0;
      for (let yy = -r; yy <= r; yy++) for (let xx = -r; xx <= r; xx++) {
        const nx = x + xx, ny = y + yy;
        if (nx < 0 || ny < 0 || nx >= padW || ny >= padH) continue;
        const wgt = Math.exp(-(xx * xx + yy * yy) / (2 * smoothPx * smoothPx));
        s += wgt * c[ny * padW + nx]; n += wgt;
      }
      out[y * padW + x] = n ? s / n : 0;
    }
  }
  return out;
}

// Layer identifiability: a layer whose coverage is tiny or whose motion does
// not reduce the data residual is not supported by the observations. The
// returned uncertainty map marks reference-canvas pixels where the layer's
// latent is effectively free (coverage below threshold or weak improvement).
export function layerUncertainty(layers, built, geom = GEOM, opts = {}) {
  const { padW, padH, obsW, obsH, ox, oy } = geom;
  const coverThr = opts.coverThr ?? 0.15;
  const maps = layers.map(() => new Float64Array(padW * padH));
  const coverage = layers.map(() => new Float64Array(padW * padH));
  for (const frame of built.perFrame) {
    frame.layers.forEach((entry, l) => {
      const m = entry.maskCanvas;
      for (let i = 0; i < padW * padH; i++) coverage[l][i] += m[i];
    });
  }
  for (let l = 0; l < layers.length; l++) {
    for (let i = 0; i < padW * padH; i++) coverage[l][i] /= built.perFrame.length;
    for (let y = 0; y < obsH; y++) {
      for (let x = 0; x < obsW; x++) {
        const i = (y + oy) * padW + (x + ox);
        maps[l][i] = coverage[l][i] < coverThr ? 1 : 1 - clamp(coverage[l][i] / coverThr, 0, 1);
      }
    }
  }
  return { maps, coverage };
}

// ---------------------------------------------------------------------------
// Velocity re-estimation by forward-residual scan.
//
// The background is re-estimated with the foreground lock region protected:
// candidate background velocities are scored only on background-covered pixels.
// The foreground candidate scan is then restricted to its moving lock region.
// ---------------------------------------------------------------------------

function compositeResidual(xCanvases, built, observations, geom, regionMasks = null) {
  const { obsW, obsH } = geom;
  const pred = frozenForward(xCanvases, built, geom);
  let se = 0, n = 0;
  for (let k = 0; k < observations.length; k++) {
    const reg = regionMasks ? regionMasks[k] : null;
    for (let i = 0; i < obsW * obsH; i++) {
      if (reg && reg[i] <= 0.05) continue;
      const d = observations[k][i] - pred[k][i];
      se += d * d; n++;
    }
  }
  return n ? Math.sqrt(se / n) : Infinity;
}

function cloneLayers(layers) {
  return layers.map((l) => ({ ...l }));
}

// Re-estimate one layer's velocity with the other layers held fixed. The scan
// is a coarse grid around the current velocity; degrees of freedom stay capped
// (only vx, vy of a linear path are searched here).
function scanLayerVelocity(layerIndex, layers, obs, geom, rho, exposureKind, xCanvases, opts) {
  const refIndex = opts.refIndex;
  const cur = layers[layerIndex];
  const base = { vx: cur.vx ?? 0, vy: cur.vy ?? 0 };
  const span = opts.scanSpan ?? 2.5;
  const step = opts.scanStep ?? 1.0;
  let best = { rmse: Infinity, vx: base.vx, vy: base.vy };
  const grid = [];
  for (let dvx = -span; dvx <= span + 1e-9; dvx += step) {
    for (let dvy = -span; dvy <= span + 1e-9; dvy += step) grid.push([base.vx + dvx, base.vy + dvy]);
  }
  for (const [vx, vy] of grid) {
    const trial = cloneLayers(layers);
    trial[layerIndex] = { ...trial[layerIndex], vx, vy };
    const built = buildLayeredOperators(trial, obs.length, refIndex, rho, exposureKind, geom, 'frozen');
    const region = layerIndex === 0
      ? null
      : built.perFrame.map((fr) => cropObservation(fr.layers[layerIndex].maskCanvas, geom.padW, geom.obsW, geom.obsH, geom.ox, geom.oy));
    // Background scan must exclude foreground pixels (protected lock region).
    let bgRegion = null;
    if (layerIndex === 0 && layers.length > 1) {
      bgRegion = built.perFrame.map((fr) => {
        const fg = fr.layers[1].maskCanvas;
        const crop = cropObservation(fg, geom.padW, geom.obsW, geom.obsH, geom.ox, geom.oy);
        return crop.map((v) => 1 - clamp(v));
      });
    }
    const rmse = compositeResidual(xCanvases, built, obs, geom, bgRegion ?? region);
    if (rmse < best.rmse) best = { rmse, vx, vy };
  }
  return best;
}

// ---------------------------------------------------------------------------
// Public pipeline entry point
// ---------------------------------------------------------------------------

// obs: array of observation frames (possibly a frameRange slice).
// spec: {
//   layers: [{ kind, vx, vy, maskCanvas? , shape? , exposureKind? }],
//   refIndex (local to the slice), rho, exposureKind, lambda, maxIter,
//   alternations, estimateMasks (bool), bgVelocity (seed), fgVelocity (seed)
// }
export function invertLayered(obs, spec, geom = GEOM) {
  const { obsW, obsH, padW, padH, ox, oy } = geom;
  const frames = obs.length;
  const refIndex = spec.refIndex ?? Math.floor(frames / 2);
  const rho = spec.rho ?? 0.8;
  const exposureKind = spec.exposureKind ?? 'uniform';
  const lambda = spec.lambda ?? 0.02;
  const maxIter = spec.maxIter ?? 35;
  const alternations = spec.alternations ?? 1;
  if (spec.layers.length > MAX_LAYERS) throw new Error(`at most ${MAX_LAYERS} motion layers`);

  let layers = cloneLayers(spec.layers);
  // Data-driven lock regions when explicit masks are not supplied.
  if (spec.estimateMasks) {
    const bgVel = spec.bgVelocity ?? { x: layers[0].vx ?? 0, y: layers[0].vy ?? 0 };
    const fg = estimateForegroundRegion(obs, bgVel, refIndex, obsW, obsH);
    const fgCanvas = obsMaskToCanvas(fg, obsW, obsH, padW, padH, ox, oy);
    const bgCanvas = new Float64Array(padW * padH);
    for (let i = 0; i < padW * padH; i++) bgCanvas[i] = 1 - clamp(fgCanvas[i]);
    // Observation window only.
    for (let y = 0; y < padH; y++) for (let x = 0; x < padW; x++) {
      const inside = x >= ox && x < ox + obsW && y >= oy && y < oy + obsH;
      if (!inside) { bgCanvas[y * padW + x] = 0; fgCanvas[y * padW + x] = 0; }
    }
    layers[0] = { ...layers[0], maskCanvas: bgCanvas };
    if (layers[1]) layers[1] = { ...layers[1], maskCanvas: fgCanvas };
    for (let l = 2; l < layers.length; l++) {
      if (!layers[l].maskCanvas) layers[l] = { ...layers[l], maskCanvas: new Float64Array(padW * padH) };
    }
  } else {
    for (const l of layers) {
      if (!l.maskCanvas) l.maskCanvas = l.shape ? shapeMask(padW, padH, l.shape) : new Float64Array(padW * padH);
    }
  }

  const velocityLog = [];
  let built = buildLayeredOperators(layers, frames, refIndex, rho, exposureKind, geom, 'frozen');
  let solve = solveLayered({ observations: obs, built, layers, geom, lambda, maxIter });

  for (let a = 0; a < alternations; a++) {
    // Re-estimate background motion with the foreground region protected, then
    // re-estimate each foreground layer inside its moving lock region.
    for (let l = 0; l < layers.length; l++) {
      const best = scanLayerVelocity(l, layers, obs, geom, rho, exposureKind, solve.xCanvases, { refIndex, ...spec });
      velocityLog.push({ alternation: a, layer: l, vx: best.vx, vy: best.vy, rmse: best.rmse });
      layers[l] = { ...layers[l], vx: best.vx, vy: best.vy };
    }
    built = buildLayeredOperators(layers, frames, refIndex, rho, exposureKind, geom, 'frozen');
    solve = solveLayered({ observations: obs, built, layers, geom, lambda, maxIter, init: solve.xCanvases });
  }

  const predicted = frozenForward(solve.xCanvases, built, geom);
  const reblur = predicted.map((pk, k) => {
    const residual = new Float64Array(obsW * obsH);
    for (let i = 0; i < residual.length; i++) residual[i] = obs[k][i] - pk[i];
    return { predicted: pk, residual };
  });
  const { maps: uncertainty, coverage } = layerUncertainty(layers, built, geom);
  // Composite estimate at the reference frame (coverage-weighted latents).
  const refFrame = built.perFrame[refIndex];
  const composite = new Float64Array(obsW * obsH);
  for (let l = 0; l < layers.length; l++) {
    const crop = cropObservation(solve.xCanvases[l], padW, obsW, obsH, ox, oy);
    const mcrop = cropObservation(refFrame.layers[l].maskCanvas, padW, obsW, obsH, ox, oy);
    for (let i = 0; i < composite.length; i++) composite[i] += mcrop[i] * crop[i];
  }
  const layerEstimates = solve.xCanvases.map((c) => cropObservation(c, padW, obsW, obsH, ox, oy));
  const refMasks = refFrame.layers.map((e) => cropObservation(e.maskCanvas, padW, obsW, obsH, ox, oy));

  return {
    layers,
    built,
    estimate: composite,
    layerEstimates,
    layerMasks: refMasks,
    uncertainty: uncertainty.map((m) => cropObservation(m, padW, obsW, obsH, ox, oy)),
    coverage: coverage.map((m) => cropObservation(m, padW, obsW, obsH, ox, oy)),
    reblur,
    predicted,
    iterations: solve.iterations,
    history: solve.history,
    velocityLog,
    strategy: 'independent',
    kernels: built.perFrame[refIndex].layers.map((e) => e.kernel),
  };
}

// Layered forward residual RMSE (optionally restricted to an obs-grid region).
export function layeredResidualRmse(result, obs, geom = GEOM, region = null) {
  let se = 0, n = 0;
  for (let k = 0; k < obs.length; k++) {
    for (let i = 0; i < obs[k].length; i++) {
      if (region && region[i] <= 0.05) continue;
      const d = result.reblur[k].residual[i];
      se += d * d; n++;
    }
  }
  return n ? Math.sqrt(se / n) : Infinity;
}

// ---------------------------------------------------------------------------
// Strategy comparison: independent per-layer kernels vs a shared background
// kernel with the foreground fitted as a residual afterwards. Both are scored
// by the SAME forward (re-blur) residual, and the lower-RMSE one is selected.
//
//   independent : every layer owns its full exposure operator and the joint
//                 stacked normal system is solved once (the default model).
//   shared-bg   : the background layer alone explains the frame; each extra
//                 layer then solves only for the residual the background cannot
//                 cover inside its lock region. Fewer coupled degrees of
//                 freedom, so it is the safer choice when the foreground is
//                 weakly identifiable.
// ---------------------------------------------------------------------------
export function compareLayeredStrategies(obs, spec, geom = GEOM) {
  // Strategy A: fully independent per-layer kernels (joint solve).
  const independent = invertLayered(obs, spec, geom);
  const rmseIndependent = layeredResidualRmse(independent, obs, geom);

  // Strategy B: shared background kernel + foreground residual fitting.
  const shared = invertSharedBackground(obs, spec, geom);
  const rmseShared = layeredResidualRmse(shared, obs, geom);

  const select = rmseIndependent <= rmseShared ? 'independent' : 'shared-bg';
  return {
    selected: select,
    scores: { independent: rmseIndependent, 'shared-bg': rmseShared },
    // Forward-RMSE gap as a fraction of the better score; a small gap means the
    // data cannot distinguish the two parameterisations and the more
    // constrained shared-bg model should be preferred on parsimony grounds.
    margin: Math.abs(rmseIndependent - rmseShared) / Math.max(Math.min(rmseIndependent, rmseShared), EPS),
    results: { independent, 'shared-bg': shared },
  };
}

// Shared-background strategy. First solve a single (background) layer over the
// whole window, then for each extra layer fit only the residual the background
// leaves inside that layer's lock region, holding the background fixed.
function invertSharedBackground(obs, spec, geom = GEOM) {
  const { obsW, obsH, padW, padH, ox, oy } = geom;
  const frames = obs.length;
  const refIndex = spec.refIndex ?? Math.floor(frames / 2);
  const rho = spec.rho ?? 0.8;
  const exposureKind = spec.exposureKind ?? 'uniform';
  const lambda = spec.lambda ?? 0.02;
  const maxIter = spec.maxIter ?? 35;
  const n = padW * padH;

  const baseLayers = cloneLayers(spec.layers);
  // Resolve lock regions exactly as the default pipeline does.
  const prepared = prepareLayers(baseLayers, obs, spec, geom);
  const bgMask = Float64Array.from(prepared[0].maskCanvas);

  // 1) Background-only solve with a full-window pinned coverage.
  const bgLayer = { ...prepared[0], maskCanvas: bgMask, fixedCoverage: true };
  const bgBuilt = buildLayeredOperators([bgLayer], frames, refIndex, rho, exposureKind, geom, 'frozen');
  const bgSolve = solveLayered({ observations: obs, built: bgBuilt, layers: [bgLayer], geom, lambda, maxIter });
  const xCanvases = [bgSolve.xCanvases[0]];
  const finalLayers = [bgLayer];

  // 2) Foreground residual layers, solved one at a time against the background
  //    prediction, restricted to each layer's moving lock region.
  const bgPred = frozenForward(xCanvases, bgBuilt, geom);
  for (let l = 1; l < prepared.length; l++) {
    const fgLayer = { ...prepared[l] };
    const fgBuilt = buildLayeredOperators([bgLayer, fgLayer], frames, refIndex, rho, exposureKind, geom, 'frozen');
    // Residual target: what the background did not explain.
    const residualObs = obs.map((g, k) => {
      const r = new Float64Array(obsW * obsH);
      for (let i = 0; i < r.length; i++) r[i] = g[i] - bgPred[k][i];
      return r;
    });
    // Single foreground layer fit on the residual target.
    const fgOnlyBuilt = buildLayeredOperators([fgLayer], frames, refIndex, rho, exposureKind, geom, 'frozen');
    const fgSolve = solveLayered({ observations: residualObs, built: fgOnlyBuilt, layers: [fgLayer], geom, lambda, maxIter });
    xCanvases.push(fgSolve.xCanvases[0]);
    finalLayers.push(fgLayer);
  }

  // 3) Re-assemble the full forward model and re-blur all layers together for a
  //    genuine closed-loop residual against the original observations.
  const built = buildLayeredOperators(finalLayers, frames, refIndex, rho, exposureKind, geom, 'frozen');
  const predicted = frozenForward(xCanvases, built, geom);
  const reblur = predicted.map((pk, k) => {
    const residual = new Float64Array(obsW * obsH);
    for (let i = 0; i < residual.length; i++) residual[i] = obs[k][i] - pk[i];
    return { predicted: pk, residual };
  });
  const { maps: uncertainty, coverage } = layerUncertainty(finalLayers, built, geom);
  const refFrame = built.perFrame[refIndex];
  const composite = new Float64Array(obsW * obsH);
  for (let l = 0; l < finalLayers.length; l++) {
    const crop = cropObservation(xCanvases[l], padW, obsW, obsH, ox, oy);
    const mcrop = cropObservation(refFrame.layers[l].maskCanvas, padW, obsW, obsH, ox, oy);
    for (let i = 0; i < composite.length; i++) composite[i] += mcrop[i] * crop[i];
  }
  return {
    layers: finalLayers,
    built,
    estimate: composite,
    layerEstimates: xCanvases.map((c) => cropObservation(c, padW, obsW, obsH, ox, oy)),
    layerMasks: refFrame.layers.map((e) => cropObservation(e.maskCanvas, padW, obsW, obsH, ox, oy)),
    uncertainty: uncertainty.map((m) => cropObservation(m, padW, obsW, obsH, ox, oy)),
    coverage: coverage.map((m) => cropObservation(m, padW, obsW, obsH, ox, oy)),
    reblur,
    predicted,
    iterations: bgSolve.iterations,
    history: bgSolve.history,
    velocityLog: [],
    strategy: 'shared-bg',
    kernels: built.perFrame[refIndex].layers.map((e) => e.kernel),
  };
}

// Resolve the per-layer coverage masks (data-driven or explicit) shared by both
// inversion strategies, mirroring the mask-preparation block in invertLayered.
function prepareLayers(layers, obs, spec, geom) {
  const { obsW, obsH, padW, padH, ox, oy } = geom;
  const refIndex = spec.refIndex ?? Math.floor(obs.length / 2);
  if (!spec.estimateMasks) {
    for (const l of layers) {
      if (!l.maskCanvas) l.maskCanvas = l.shape ? shapeMask(padW, padH, l.shape) : new Float64Array(padW * padH);
    }
    return layers;
  }
  const bgVel = spec.bgVelocity ?? { x: layers[0].vx ?? 0, y: layers[0].vy ?? 0 };
  const fg = estimateForegroundRegion(obs, bgVel, refIndex, obsW, obsH);
  const fgCanvas = obsMaskToCanvas(fg, obsW, obsH, padW, padH, ox, oy);
  const bgCanvas = new Float64Array(padW * padH);
  for (let i = 0; i < padW * padH; i++) bgCanvas[i] = 1 - clamp(fgCanvas[i]);
  for (let y = 0; y < padH; y++) for (let x = 0; x < padW; x++) {
    const inside = x >= ox && x < ox + obsW && y >= oy && y < oy + obsH;
    if (!inside) { bgCanvas[y * padW + x] = 0; fgCanvas[y * padW + x] = 0; }
  }
  layers[0] = { ...layers[0], maskCanvas: bgCanvas, fixedCoverage: true };
  if (layers[1]) layers[1] = { ...layers[1], maskCanvas: fgCanvas };
  for (let l = 2; l < layers.length; l++) {
    if (!layers[l].maskCanvas) layers[l] = { ...layers[l], maskCanvas: new Float64Array(padW * padH) };
  }
  return layers;
}

// ---------------------------------------------------------------------------
// Synthetic layered clip generator (shared discrete model; truth stays separate)
// ---------------------------------------------------------------------------

// Add a foreground object to a scene: an elliptical patch with its own texture,
// painted at a reference position. Returns the modified scene plus truth
// geometry used to build the foreground latent and its coverage shape.
export function addForegroundObject(scene, opts = {}) {
  const { w, h } = scene;
  const cx = opts.cx ?? w * 0.32;
  const cy = opts.cy ?? h * 0.55;
  const rx = opts.rx ?? 34;
  const ry = opts.ry ?? 26;
  const angle = opts.angleRad ?? 0.18;
  const tone = opts.tone ?? 0.25;
  const c = Math.cos(-angle), s = Math.sin(-angle);
  const data = Float64Array.from(scene.data);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
      const lx = dx * c - dy * s, ly = dx * s + dy * c;
      const inside = (lx * lx) / (rx * rx) + (ly * ly) / (ry * ry) <= 1;
      if (!inside) continue;
      const stripe = 0.5 + 0.5 * Math.sin(lx * 0.55 + 1.7 * Math.sin(ly * 0.3));
      const edge = 1 - ((lx * lx) / (rx * rx) + (ly * ly) / (ry * ry));
      data[y * w + x] = clamp(0.18 + tone * stripe + 0.12 * edge);
    }
  }
  return {
    scene: { data, w, h },
    object: { cx, cy, rx, ry, angleRad: angle },
  };
}

// makeLayeredClip: background follows bgMotion, one foreground object follows
// fgMotion (in obs px/frame). The exact time-sampled composite forward produces
// the observations; the truth latents and masks are returned separately and
// never reach the solver.
export function makeLayeredClip({
  seed = 21,
  bgMotion = { type: 'pan', params: { vx: 4.0, vy: 0.6 } },
  fgMotion = { vx: -5.0, vy: -1.6 },
  rho = 0.8,
  exposureKind = 'uniform',
  frames = 7,
  refIndex = 3,
  noiseSigma = 0.004,
  quantize = true,
  exposureSamples = 24,
  fgOpts = {},
  // 'exact' re-evaluates coverage at every exposure sample (boundaries move
  // during the shutter); 'frozen' freezes coverage at the frame centre, the
  // same approximation the default inversion uses. Exact is the physical
  // model; frozen clips let generator and inversion close the loop exactly.
  boundaryMode = 'exact',
} = {}) {
  const { obsW, obsH, padW, padH, ox, oy, sceneW, sceneH } = GEOM;
  const baseScene = makeScene(seed);
  const { scene, object } = addForegroundObject(baseScene, fgOpts);

  // Background latent: area-average the scene, then remove the foreground
  // footprint so the background layer has a clean canvas (inpainting by local
  // median of surrounding pixels; only used for the synthetic truth).
  const fullLatent = latentFromScene(scene);
  const bgLatent = inpaintForeground(fullLatent, object, obsW, obsH, sceneW, sceneH);
  // Foreground latent: full latent inside the object, zero elsewhere.
  const fgLatent = new Float64Array(obsW * obsH);
  const fgShapeObs = {
    cx: object.cx * (obsW / sceneW) + ox,
    cy: object.cy * (obsH / sceneH) + oy,
    rx: object.rx * (obsW / sceneW),
    ry: object.ry * (obsH / sceneH),
    angleRad: object.angleRad,
  };
  const fgMaskCanvas = shapeMask(padW, padH, fgShapeObs, 1.0);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      const i = y * obsW + x;
      const ci = (y + oy) * padW + (x + ox);
      fgLatent[i] = fullLatent[i] * fgMaskCanvas[ci];
    }
  }
  // Complementary coverage: the foreground mask m and the background mask 1-m
  // partition every observed pixel (a proper alpha composite), so the layers
  // have no shared null space. Overlapping all-ones + m coverage would let the
  // two latents cancel inside the object and make the joint system singular.
  const bgMaskCanvas = new Float64Array(padW * padH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      const i = (y + oy) * padW + (x + ox);
      bgMaskCanvas[i] = 1 - fgMaskCanvas[i];
    }
  }

  const bgPath = makeParametricPath({
    kind: bgMotion.type === 'shake' ? 'shake' : 'linear',
    vx: bgMotion.params.vx ?? 0, vy: bgMotion.params.vy ?? 0,
    amp: bgMotion.params.amp ?? 0, freq: bgMotion.params.freq ?? 0,
  });
  const fgPath = makeParametricPath({ kind: 'linear', vx: fgMotion.vx, vy: fgMotion.vy });
  const layers = [
    { kind: 'linear', vx: bgMotion.params.vx ?? 0, vy: bgMotion.params.vy ?? 0, maskCanvas: bgMaskCanvas, exposureKind, fixedCoverage: true },
    { kind: 'linear', vx: fgMotion.vx, vy: fgMotion.vy, maskCanvas: fgMaskCanvas, exposureKind },
  ];
  const paths = [bgPath, fgPath];
  const bgCanvas = embedObservation(bgLatent, obsW, obsH, padW, padH, ox, oy);
  const fgCanvas = embedObservation(fgLatent, obsW, obsH, padW, padH, ox, oy);

  const obs = [];
  const clean = [];
  const frozenBuilt = boundaryMode === 'frozen'
    ? buildLayeredOperators(layers, frames, refIndex, rho, exposureKind, GEOM, 'frozen')
    : null;
  const frozenPred = frozenBuilt
    ? frozenForward([bgCanvas, fgCanvas], frozenBuilt, GEOM)
    : null;
  for (let k = 0; k < frames; k++) {
    const gFull = boundaryMode === 'frozen'
      ? frozenPred[k]
      : cropObservation(
          exactFrameForward([bgCanvas, fgCanvas], layers, paths, k, refIndex, rho, exposureKind, padW, padH, exposureSamples),
          padW, obsW, obsH, ox, oy);
    let g = gFull;
    if (quantize) g = quantize8(g);
    g = addNoise(g, noiseSigma, seed * 131 + k);
    obs.push(g);
    // Sharp composite at the frame centre (truth clean sequence).
    const cleanFrame = new Float64Array(obsW * obsH);
    for (let l = 0; l < 2; l++) {
      const pr = paths[l](refIndex), pk = paths[l](k);
      const sdx = pk.x - pr.x, sdy = pk.y - pr.y;
      const moved = shiftImageSpatial(l === 0 ? bgCanvas : fgCanvas, padW, padH, sdx, sdy);
      const cover = shiftMask(layers[l].maskCanvas, padW, padH, sdx, sdy);
      const mc = cropObservation(moved, padW, obsW, obsH, ox, oy);
      const cc = cropObservation(cover, padW, obsW, obsH, ox, oy);
      for (let i = 0; i < cleanFrame.length; i++) cleanFrame[i] += cc[i] * mc[i];
    }
    clean.push(cleanFrame);
  }

  return {
    obs,
    clean,
    refIndex,
    bgLatent,
    fgLatent,
    fullLatent,
    fgShapeObs,
    truthMasks: {
      bg: cropObservation(bgMaskCanvas, padW, obsW, obsH, ox, oy),
      fg: cropObservation(fgMaskCanvas, padW, obsW, obsH, ox, oy),
    },
    truthVelocities: {
      bg: { vx: bgMotion.params.vx ?? 0, vy: bgMotion.params.vy ?? 0 },
      fg: { vx: fgMotion.vx, vy: fgMotion.vy },
    },
    spec: { seed, bgMotion, fgMotion, rho, exposureKind, frames, refIndex, noiseSigma, quantize, exposureSamples, boundaryMode },
  };
}

// Replace the foreground footprint of a latent with a local median of the ring
// just outside the object, so the background truth layer is a complete image.
function inpaintForeground(latent, object, obsW, obsH, sceneW, sceneH) {
  const out = Float64Array.from(latent);
  const sx = obsW / sceneW, sy = obsH / sceneH;
  const cx = object.cx * sx, cy = object.cy * sy;
  const rx = object.rx * sx + 2, ry = object.ry * sy + 2;
  const c = Math.cos(-object.angleRad), s = Math.sin(-object.angleRad);
  const ring = [];
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
      const lx = dx * c - dy * s, ly = dx * s + dy * c;
      const q = (lx * lx) / (rx * rx) + (ly * ly) / (ry * ry);
      if (q <= 1) out[y * obsW + x] = NaN;
      else if (q <= 1.6) ring.push(latent[y * obsW + x]);
    }
  }
  ring.sort((a, b) => a - b);
  const fill = ring.length ? ring[ring.length >> 1] : 0.4;
  for (let i = 0; i < out.length; i++) if (Number.isNaN(out[i])) out[i] = fill;
  return out;
}

// Single-motion model as a one-layer special case of the layered machinery, so
// the global-motion path remains a runnable configuration and shares the same
// forward verification.
export function invertSingleLayer(obs, spec, geom = GEOM) {
  const { padW, padH, obsW, obsH, ox, oy } = geom;
  const mask = new Float64Array(padW * padH);
  for (let y = 0; y < obsH; y++) for (let x = 0; x < obsW; x++) mask[(y + oy) * padW + (x + ox)] = 1;
  return invertLayered(obs, {
    ...spec,
    estimateMasks: false,
    layers: [{ kind: spec.kind ?? 'linear', vx: spec.vx ?? 0, vy: spec.vy ?? 0, maskCanvas: mask, exposureKind: spec.exposureKind }],
  }, geom);
}
