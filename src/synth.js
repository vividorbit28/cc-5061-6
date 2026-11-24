// Synthetic scene + observation generator.
//
// Both the generator and the inversion use the *same* discrete model:
//
//   x_ref : sharp latent on the observation grid (area-averaged from a 2x scene)
//   h_k   : exposure kernel of frame k, path-integrated over its shutter
//   d_k   : displacement of frame k relative to the reference frame
//   g_k   = crop( S_{d_k} ( h_k * x_ref ) )
//
// where S is an exact Fourier sub-pixel shift and * is circular convolution on
// the padded canvas. Ground truth (x_ref and the clean sequence) is returned
// separately; the inversion never receives it. Because generation and inversion
// share the operators, "re-blur the estimate" is a genuine closed-loop test:
// with the right kernel the residual falls to the noise level, with a wrong one
// it does not.

import {
  mulberry32, clamp, smoothstep, hash2, blurSeparable,
} from './util.js';
import { embedObservation, cropObservation, forwardAll, buildFrameKernels, operatorSpectrum } from './model.js';
import { shiftCanvasSpatial } from './model.js';

export const GEOM = {
  sceneW: 320,
  sceneH: 192,
  obsW: 160,
  obsH: 96,
  padW: 256,
  padH: 128,
  ox: 48,
  oy: 16,
};

export const DEFAULTS = {
  frames: 7,
  refIndex: 3,
  rho: 0.8,
  noiseSigma: 0.004,
  quantize8bit: true,
  exposureSamples: 33,
};

// ---------------------------------------------------------------------------
// Scene texture (rendered at 2x, area-averaged down to the observation grid)
// ---------------------------------------------------------------------------

function valueNoise(seed) {
  return (x, y) => {
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const tx = smoothstep(x - x0), ty = smoothstep(y - y0);
    const a = hash2(x0, y0, seed), b = hash2(x0 + 1, y0, seed);
    const c = hash2(x0, y0 + 1, seed), d = hash2(x0 + 1, y0 + 1, seed);
    const ab = a + (b - a) * tx;
    const cd = c + (d - c) * tx;
    return ab + (cd - ab) * ty;
  };
}

function fbm(n, x, y, octaves, gain = 0.5, lac = 2.03) {
  let amp = 1, freq = 1, sum = 0, norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * n(x * freq + o * 17.3, y * freq - o * 9.7);
    norm += amp;
    amp *= gain;
    freq *= lac;
  }
  return sum / norm;
}

function sdRoundRect(px, py, cx, cy, hw, hh, r) {
  const qx = Math.abs(px - cx) - hw + r;
  const qy = Math.abs(py - cy) - hh + r;
  const ax = Math.max(qx, 0), ay = Math.max(qy, 0);
  return Math.sqrt(ax * ax + ay * ay) + Math.min(Math.max(qx, qy), 0) - r;
}

function sdCircle(px, py, cx, cy, r) {
  return Math.hypot(px - cx, py - cy) - r;
}

function barChart(px, py, cx, cy, n, period, halfH) {
  const lx = px - cx, ly = py - cy;
  if (Math.abs(ly) > halfH || Math.abs(lx) > (n * period) / 2) return 0;
  const idx = Math.floor((lx + (n * period) / 2) / period);
  return idx % 2 === 0 ? 1 : 0;
}

export function makeScene(seed = 7) {
  const W = GEOM.sceneW;
  const H = GEOM.sceneH;
  const out = new Float64Array(W * H);
  const n1 = valueNoise(seed * 3 + 1);
  const n2 = valueNoise(seed * 7 + 2);
  const n3 = valueNoise(seed * 11 + 3);
  const rand = mulberry32(seed);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W, v = y / H;
      out[y * W + x] =
        0.22 +
        0.16 * (0.5 + 0.5 * Math.sin(2 * Math.PI * (0.7 * u + 0.2 * v) + 1.1)) +
        0.10 * fbm(n1, u * 3.1, v * 3.1, 4);
    }
  }

  const shapes = [];
  for (let i = 0; i < 30; i++) {
    shapes.push({
      x: rand() * W, y: rand() * H,
      hw: 8 + rand() * 26, hh: 5 + rand() * 16,
      rot: rand() * Math.PI, tone: (rand() - 0.5) * 0.26,
      round: rand() > 0.5,
    });
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = x + 0.5, py = y + 0.5;
      let add = 0;
      for (const s of shapes) {
        const dx = px - s.x, dy = py - s.y;
        const c = Math.cos(-s.rot), sn = Math.sin(-s.rot);
        const lx = dx * c - dy * sn, ly = dx * sn + dy * c;
        const d = s.round
          ? Math.hypot(lx, ly) - Math.min(s.hw, s.hh) * 0.6
          : sdRoundRect(lx, ly, 0, 0, s.hw, s.hh, 3.5);
        add += clamp(0.5 - d * 1.1) * s.tone;
      }
      out[y * W + x] += add * 0.5;
    }
  }

  // Bar chart: 6 bars, 9 scene px pitch -> 4.5 obs px (clearly recoverable).
  const chart = { x: W * 0.70, y: H * 0.34, n: 6, period: 9, halfH: 20 };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const b = barChart(x + 0.5, y + 0.5, chart.x, chart.y, chart.n, chart.period, chart.halfH);
      if (b) out[y * W + x] = 0.88 - 0.05 * fbm(n2, x / 9, y / 9, 2);
    }
  }

  // Fine diagonal texture patch: ~1.5 obs px period, at/above the observation
  // Nyquist limit. Under a long exposure this is not recoverable; it is used to
  // show where the result is prior fill rather than measured detail.
  const texCx = W * 0.27, texCy = H * 0.67, texS = 30;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d = Math.max(Math.abs(x - texCx), Math.abs(y - texCy)) - texS;
      const a = clamp(0.5 - d / 2.5);
      if (a <= 0) continue;
      const s = 0.5 + 0.5 * Math.sin((x * 0.5 + y) * 1.9 + 0.4 * fbm(n3, x / 6, y / 6, 2));
      out[y * W + x] = out[y * W + x] * (1 - a) + a * (0.30 + 0.34 * s);
    }
  }

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const px = x + 0.5, py = y + 0.5;
      const d1 = sdCircle(px, py, W * 0.55, H * 0.72, 22);
      const d2 = sdRoundRect(px, py, W * 0.44, H * 0.28, 34, 14, 5);
      const a1 = clamp(0.5 - d1);
      const a2 = clamp(0.5 - d2);
      if (a1 > 0) out[y * W + x] += a1 * 0.17;
      if (a2 > 0) out[y * W + x] -= a2 * 0.10;
    }
  }

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W - 0.5;
      const v = (y + 0.5) / H - 0.5;
      const vig = 1 - 0.20 * (u * u + v * v) * 2;
      out[y * W + x] = clamp(0.02 + 0.96 * out[y * W + x] * vig);
    }
  }
  return { data: out, w: W, h: H };
}

// Area-average the 2x scene onto the observation grid. This is the sharp latent
// at the reference time; it is the evaluation target and never an inversion input.
export function latentFromScene(scene) {
  const { obsW, obsH, sceneW, sceneH } = GEOM;
  const sx = sceneW / obsW;
  const sy = sceneH / obsH;
  const out = new Float64Array(obsW * obsH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      let s = 0, n = 0;
      const x0 = Math.floor(x * sx), x1 = Math.floor((x + 1) * sx);
      const y0 = Math.floor(y * sy), y1 = Math.floor((y + 1) * sy);
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          s += scene.data[yy * scene.w + xx];
          n++;
        }
      }
      out[y * obsW + x] = s / n;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Motion paths (observation pixels per frame interval)
// ---------------------------------------------------------------------------

export const MOTION_PRESETS = {
  pan: { label: 'constant pan', params: () => ({ vx: 7.0, vy: 1.2 }) },
  diag: { label: 'diagonal pan', params: () => ({ vx: 6.5, vy: 3.8 }) },
  fast: { label: 'fast pan (strong blur)', params: () => ({ vx: 11.0, vy: 0.5 }) },
  shake: {
    label: 'pan + vertical shake',
    params: () => ({ vx: 5.5, vy: 0.4, amp: 2.2, freq: 0.9 }),
  },
  arc: {
    label: 'shutter arc (rotation)',
    params: () => ({ omega: 0.03, pivotX: -120, pivotY: -60 }),
  },
  stationary: { label: 'stationary', params: () => ({}) },
};

// Position along the path at time tau (frames). tau = k is the centre of the
// exposure of frame k.
export function motionAt(type, params, tau) {
  switch (type) {
    case 'pan':
    case 'diag':
      return { x: params.vx * tau, y: params.vy * tau };
    case 'fast':
      return { x: params.vx * tau, y: params.vy * tau };
    case 'shake':
      return {
        x: params.vx * tau,
        y: params.vy * tau + params.amp * Math.sin(2 * Math.PI * params.freq * tau),
      };
    case 'arc': {
      const a = params.omega * tau;
      const px = GEOM.sceneW / 2, py = GEOM.sceneH / 2;
      const dx = px - params.pivotX, dy = py - params.pivotY;
      const nx = params.pivotX + dx * Math.cos(a) - dy * Math.sin(a);
      const ny = params.pivotY + dx * Math.sin(a) + dy * Math.cos(a);
      return { x: (nx - px) / 2, y: (ny - py) / 2 };
    }
    case 'stationary':
    default:
      return { x: 0, y: 0 };
  }
}

export function pathFromMotion(motion) {
  return (tau) => motionAt(motion.type, motion.params, tau);
}

// Parametric path used for hypotheses (linear, shake, arc) without needing the
// generator's preset table.
export function makeParametricPath(spec) {
  const { kind = 'linear', vx = 0, vy = 0, amp = 0, freq = 0, omega = 0, pivotX = 0, pivotY = 0 } = spec;
  return (tau) => {
    switch (kind) {
      case 'shake':
        return { x: vx * tau, y: vy * tau + amp * Math.sin(2 * Math.PI * freq * tau) };
      case 'arc': {
        const a = omega * tau;
        const px = GEOM.obsW / 2, py = GEOM.obsH / 2;
        const dx = px - pivotX, dy = py - pivotY;
        const nx = pivotX + dx * Math.cos(a) - dy * Math.sin(a);
        const ny = pivotY + dx * Math.sin(a) + dy * Math.cos(a);
        return { x: nx - px, y: ny - py };
      }
      case 'linear':
      default:
        return { x: vx * tau, y: vy * tau };
    }
  };
}

export function exposureWeights(kind, n) {
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

// ---------------------------------------------------------------------------
// Clip generation through the shared forward model
// ---------------------------------------------------------------------------

export function quantize8(img) {
  const out = new Float64Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = Math.round(clamp(img[i]) * 255) / 255;
  return out;
}

export function addNoise(img, sigma, seed = 1) {
  if (!sigma) return Float64Array.from(img);
  const r = mulberry32(seed);
  const out = new Float64Array(img.length);
  for (let i = 0; i < img.length; i++) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    out[i] = clamp(img[i] + g * sigma);
  }
  return out;
}

export function buildOperators(pathFn, frames, refIndex, rho, exposureKind, radius, opts = {}) {
  const { padW, padH } = GEOM;
  const kernels = buildFrameKernels(pathFn, frames, rho, exposureKind, radius);
  const positions = Array.from({ length: frames }, (_, k) => pathFn(k));
  const pRef = positions[refIndex];
  const operators = kernels.map((kern, k) => {
    const dx = positions[k].x - pRef.x;
    const dy = positions[k].y - pRef.y;
    return operatorSpectrum(kern, dx, dy, padW, padH);
  });
  return { kernels, operators, displacements: positions.map((p) => ({ dx: p.x - pRef.x, dy: p.y - pRef.y })), pRef };
}

// Create a clip. Returns observations (what the inversion sees) plus truth and
// the exact operator set used to produce them (for ground-truth-forward checks).
export function makeClip({
  seed = 7,
  motionType = 'pan',
  exposureKind = 'uniform',
  frames = DEFAULTS.frames,
  refIndex = DEFAULTS.refIndex,
  rho = DEFAULTS.rho,
  noiseSigma = DEFAULTS.noiseSigma,
  quantize = DEFAULTS.quantize8bit,
  exposureSamples = DEFAULTS.exposureSamples,
  kernelRadius = null,
  motionParams = null,
  pathFn = null,
} = {}) {
  const { obsW, obsH, padW, padH, ox, oy } = GEOM;
  const scene = makeScene(seed);
  const preset = MOTION_PRESETS[motionType] ?? MOTION_PRESETS.pan;
  const params = motionParams ?? preset.params(seed);
  const motion = { type: motionType, params };
  const path = pathFn ?? pathFromMotion(motion);
  const latent = latentFromScene(scene);
  const radius = kernelRadius ?? Math.max(6, 12 + Math.hypot(params.vx ?? 0, params.vy ?? 0) * rho);
  const built = buildOperators(path, frames, refIndex, rho, exposureKind, radius, { samples: exposureSamples });

  const latentCanvas = embedObservation(latent, obsW, obsH, padW, padH, ox, oy);
  const predCanvas = forwardAll(latentCanvas, built.operators, padW, padH);
  const obs = [];
  const clean = [];
  for (let k = 0; k < frames; k++) {
    let g = cropObservation(predCanvas[k], padW, obsW, obsH, ox, oy);
    if (quantize) g = quantize8(g);
    g = addNoise(g, noiseSigma, seed * 101 + k);
    obs.push(g);
    // Sharp frame at the same path position: crop(S_{d_k} x_ref).
    const d = built.displacements[k];
    const shifted = shiftCanvasSpatial(latentCanvas, padW, padH, d.dx, d.dy);
    clean.push(cropObservation(shifted, padW, obsW, obsH, ox, oy));
  }

  return {
    scene, latent, motion, obs, clean, refIndex,
    refSharp: clean[refIndex],
    truthOperators: built.operators,
    truthKernels: built.kernels,
    truthDisplacements: built.displacements,
    spec: {
      seed, motionType, exposureKind, frames, refIndex, rho, noiseSigma, quantize,
      exposureSamples, kernelRadius: radius, motionParams: params,
    },
  };
}

// Apply a clip's truth operator set to an arbitrary latent (used to test that
// synthesis and inversion close the loop).
export function forwardFromLatent(latent, operators) {
  const { obsW, obsH, padW, padH, ox, oy } = GEOM;
  const canvas = embedObservation(latent, obsW, obsH, padW, padH, ox, oy);
  const pred = forwardAll(canvas, operators, padW, padH);
  return pred.map((p) => cropObservation(p, padW, obsW, obsH, ox, oy));
}

export { blurSeparable };
