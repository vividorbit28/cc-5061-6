// Discrete forward model shared by synthesis and inversion:
//
//   g_k = S_{d_k}( h_k * x )      (crop to the observation window)
//
// x is the sharp latent defined on a padded canvas; h_k is the exposure kernel
// (path integration over the shutter, in observation pixels); d_k is the image
// displacement of frame k relative to the reference frame; S_d is an exact
// Fourier sub-pixel shift. Both the generator and the solver use these routines,
// so "re-blur the estimate and compare with the input" is a closed loop.

import { FFT2, frequencies } from './fft.js';
import { exposureWeightsLocal, buildKernel } from './motion.js';

export function kernelToCanvas(kernel, padW, padH) {
  const c = new Float64Array(padW * padH);
  const { data, size } = kernel;
  const r = (size - 1) >> 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const yy = ((y - r) % padH + padH) % padH;
      const xx = ((x - r) % padW + padW) % padW;
      c[yy * padW + xx] = data[y * size + x];
    }
  }
  return c;
}

// Complex multiplier H(f) * exp(-i 2pi (fx dx / padW + fy dy / padH)).
export function operatorSpectrum(kernel, dx, dy, padW, padH) {
  const kc = kernelToCanvas(kernel, padW, padH);
  const f = new FFT2(padW, padH);
  f.re.set(kc);
  f.forward();
  const { fx, fy } = frequencies(padW, padH);
  const re = new Float64Array(padW * padH);
  const im = new Float64Array(padW * padH);
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const i = y * padW + x;
      const ang = -2 * Math.PI * (fx[x] * dx / padW + fy[y] * dy / padH);
      const pc = Math.cos(ang), ps = Math.sin(ang);
      const hr = f.re[i], hi = f.im[i];
      re[i] = hr * pc - hi * ps;
      im[i] = hr * ps + hi * pc;
    }
  }
  return { re, im };
}

export function embedObservation(obs, obsW, obsH, padW, padH, ox, oy) {
  const out = new Float64Array(padW * padH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) out[(y + oy) * padW + (x + ox)] = obs[y * obsW + x];
  }
  return out;
}

export function cropObservation(canvas, padW, obsW, obsH, ox, oy) {
  const out = new Float64Array(obsW * obsH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) out[y * obsW + x] = canvas[(y + oy) * padW + (x + ox)];
  }
  return out;
}

export function shiftCanvasSpatial(src, padW, padH, dx, dy) {
  const f = new FFT2(padW, padH);
  f.re.set(src);
  f.forward();
  const { fx, fy } = frequencies(padW, padH);
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const i = y * padW + x;
      const ang = -2 * Math.PI * (fx[x] * dx / padW + fy[y] * dy / padH);
      const c = Math.cos(ang), s = Math.sin(ang);
      const a = f.re[i], b = f.im[i];
      f.re[i] = a * c - b * s;
      f.im[i] = a * s + b * c;
    }
  }
  f.inverse();
  return f.re;
}

// Kernel for one frame: integrate the exposure path relative to the frame time.
// pathFn(tau) returns the image displacement in observation pixels at path time
// tau (frames). rho is the shutter fraction, kind the exposure weighting.
export function frameKernel(pathFn, k, rho, kind, radius, samples = 48) {
  const n = Math.max(8, samples);
  const w = exposureWeightsLocal(kind, n);
  const pts = new Float64Array(n * 3);
  const c = pathFn(k);
  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    const u = n === 1 ? 0.5 : i / (n - 1);
    const tau = k + (u - 0.5) * rho;
    const p = pathFn(tau);
    pts[i * 3] = p.x - c.x;
    pts[i * 3 + 1] = p.y - c.y;
    pts[i * 3 + 2] = w[i];
    maxAbs = Math.max(maxAbs, Math.hypot(pts[i * 3], pts[i * 3 + 1]));
  }
  const r = Math.max(2, Math.ceil(Math.max(radius, maxAbs / 2 + 1.5)));
  const size = 2 * r + 1;
  const data = new Float64Array(size * size);
  for (let i = 0; i < n; i++) {
    const px = pts[i * 3] + r;
    const py = pts[i * 3 + 1] + r;
    const x0 = Math.floor(px), y0 = Math.floor(py);
    const tx = px - x0, ty = py - y0;
    const wgt = pts[i * 3 + 2];
    const splat = (xi, yi, v) => {
      if (xi >= 0 && xi < size && yi >= 0 && yi < size) data[yi * size + xi] += v;
    };
    splat(x0, y0, wgt * (1 - tx) * (1 - ty));
    splat(x0 + 1, y0, wgt * tx * (1 - ty));
    splat(x0, y0 + 1, wgt * (1 - tx) * ty);
    splat(x0 + 1, y0 + 1, wgt * tx * ty);
  }
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i];
  if (s > 0) for (let i = 0; i < data.length; i++) data[i] /= s;
  return { data, size, radius: r };
}

// Build all frame kernels from a path model. rho/kind may be per-frame arrays.
export function buildFrameKernels(pathFn, frames, rho, kind, radius) {
  const kernels = [];
  for (let k = 0; k < frames; k++) {
    const r = Array.isArray(rho) ? rho[k] : rho;
    const kk = Array.isArray(kind) ? kind[k] : kind;
    kernels.push(frameKernel(pathFn, k, r, kk, radius));
  }
  return kernels;
}

// Apply the forward model to a latent canvas; returns per-frame observations on
// the padded canvas (before cropping).
export function forwardAll(xCanvas, operators, padW, padH) {
  const f = new FFT2(padW, padH);
  f.re.set(xCanvas);
  f.forward();
  const Xr = Float64Array.from(f.re);
  const Xi = Float64Array.from(f.im);
  const out = [];
  for (const op of operators) {
    for (let i = 0; i < padW * padH; i++) {
      f.re[i] = Xr[i] * op.re[i] - Xi[i] * op.im[i];
      f.im[i] = Xr[i] * op.im[i] + Xi[i] * op.re[i];
    }
    f.inverse();
    out.push(Float64Array.from(f.re));
  }
  return out;
}

export { buildKernel };
