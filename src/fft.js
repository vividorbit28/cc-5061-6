// Power-of-two FFT utilities used by the blur-inversion kernel.
// All transforms use the convention FFT: X[f] = sum_x x[x] exp(-i 2pi f x / N),
// IFFT: x[x] = (1/N) sum_f X[f] exp(+i 2pi f x / N).

const plans = new Map();

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function makePlan(n) {
  let plan = plans.get(n);
  if (plan) return plan;
  const levels = Math.log2(n) | 0;
  if (1 << levels !== n) throw new Error(`FFT length must be a power of two, got ${n}`);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i;
    let r = 0;
    for (let j = 0; j < levels; j++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  plan = { n, levels, rev, cos, sin };
  plans.set(n, plan);
  return plan;
}

function fft1d(re, im, offset, stride, n, inverse) {
  const { rev, cos, sin } = makePlan(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      const a = offset + i * stride;
      const b = offset + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step];
        const wi = inverse ? sin[k * step] : -sin[k * step];
        const a = offset + (start + k) * stride;
        const b = offset + (start + k + half) * stride;
        const br = re[b] * wr - im[b] * wi;
        const bi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - br;
        im[b] = im[a] - bi;
        re[a] += br;
        im[a] += bi;
      }
    }
  }
}

export class FFT2 {
  constructor(w, h) {
    this.w = w;
    this.h = h;
    this.n = w * h;
    this.re = new Float64Array(this.n);
    this.im = new Float64Array(this.n);
    const pw = nextPow2(w);
    const ph = nextPow2(h);
    if (pw !== w || ph !== h) throw new Error(`FFT2 size must be powers of two, got ${w}x${h}`);
  }

  forward() {
    const { w, h, re, im } = this;
    for (let y = 0; y < h; y++) fft1d(re, im, y * w, 1, w, false);
    for (let x = 0; x < w; x++) fft1d(re, im, x, w, h, false);
    return this;
  }

  inverse() {
    const { w, h, re, im } = this;
    for (let y = 0; y < h; y++) fft1d(re, im, y * w, 1, w, true);
    for (let x = 0; x < w; x++) fft1d(re, im, x, w, h, true);
    const s = 1 / this.n;
    for (let i = 0; i < this.n; i++) { re[i] *= s; im[i] *= s; }
    return this;
  }
}

export function fft2d(re, im, w, h, inverse = false) {
  if (inverse) {
    for (let y = 0; y < h; y++) fft1d(re, im, y * w, 1, w, true);
    for (let x = 0; x < w; x++) fft1d(re, im, x, w, h, true);
    const s = 1 / (w * h);
    for (let i = 0; i < w * h; i++) { re[i] *= s; im[i] *= s; }
  } else {
    for (let y = 0; y < h; y++) fft1d(re, im, y * w, 1, w, false);
    for (let x = 0; x < w; x++) fft1d(re, im, x, w, h, false);
  }
}

export function padToPow2(w, h) {
  return [nextPow2(w), nextPow2(h)];
}

// Reflect-pad a single-channel image so linear convolution / translation can be
// applied in the Fourier domain without circular wraparound.
export function reflectPad(img, w, h, margin) {
  const pw = w + 2 * margin;
  const ph = h + 2 * margin;
  const out = new Float64Array(pw * ph);
  const rr = (v, n) => {
    if (n === 1) return 0;
    const period = 2 * n - 2;
    let x = v % period;
    if (x < 0) x += period;
    return x < n ? x : period - x;
  };
  for (let y = 0; y < ph; y++) {
    const sy = rr(y - margin, h);
    for (let x = 0; x < pw; x++) {
      const sx = rr(x - margin, w);
      out[y * pw + x] = img[sy * w + sx];
    }
  }
  return { data: out, w: pw, h: ph };
}

export function cropFrame(img, pw, ph, w, h, margin) {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) out[y * w + x] = img[(y + margin) * pw + (x + margin)];
  }
  return out;
}

// Circular (or, with enough reflection padding, linear) translation by dx,dy.
// Positive dx moves image content to the right.
export function fourierShift(img, w, h, dx, dy, opts = {}) {
  const margin = opts.margin ?? 0;
  const padW = opts.padW ?? w;
  const padH = opts.padH ?? h;
  let src = img;
  if (margin > 0) {
    const p = reflectPad(img, w, h, margin);
    src = p.data;
  }
  const re = new Float64Array(src.length);
  const im = new Float64Array(src.length);
  re.set(src);
  fft2d(re, im, padW, padH, false);
  const fx = new Float64Array(padW);
  const fy = new Float64Array(padH);
  for (let x = 0; x < padW; x++) fx[x] = x < padW / 2 ? x : x - padW;
  for (let y = 0; y < padH; y++) fy[y] = y < padH / 2 ? y : y - padH;
  for (let y = 0; y < padH; y++) {
    for (let x = 0; x < padW; x++) {
      const ang = -2 * Math.PI * (fx[x] * dx / padW + fy[y] * dy / padH);
      const c = Math.cos(ang), s = Math.sin(ang);
      const i = y * padW + x;
      const a = re[i], b = im[i];
      re[i] = a * c - b * s;
      im[i] = a * s + b * c;
    }
  }
  fft2d(re, im, padW, padH, true);
  if (margin > 0) {
    return cropFrame(re, padW, padH, w, h, margin);
  }
  return re;
}

// Sinusoidal window (Hann) for phase correlation.
export function hann2d(w, h, edge = 0.08) {
  const out = new Float64Array(w * h);
  const wx = new Float64Array(w);
  const wy = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    const t = x / (w - 1);
    const e = Math.min(1, Math.min(t, 1 - t) / edge);
    wx[x] = e * e * (3 - 2 * e);
  }
  for (let y = 0; y < h; y++) {
    const t = y / (h - 1);
    const e = Math.min(1, Math.min(t, 1 - t) / edge);
    wy[y] = e * e * (3 - 2 * e);
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = wx[x] * wy[y];
  return out;
}

export { nextPow2 };

// ---------------------------------------------------------------------------
// Grid helpers for the padded observation model.
// ---------------------------------------------------------------------------

export function frequencies(w, h) {
  const fx = new Float64Array(w);
  const fy = new Float64Array(h);
  for (let x = 0; x < w; x++) fx[x] = x < w / 2 ? x : x - w;
  for (let y = 0; y < h; y++) fy[y] = y < h / 2 ? y : y - h;
  return { fx, fy };
}

export function cropAt(src, sw, ox, oy, cw, ch) {
  const out = new Float64Array(cw * ch);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) out[y * cw + x] = src[(y + oy) * sw + (x + ox)];
  }
  return out;
}

export function padAt(dst, dw, src, cw, ch, ox, oy) {
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) dst[(y + oy) * dw + (x + ox)] = src[y * cw + x];
  }
}

// Shift content by (dx,dy) on a periodic padded grid. Positive dx moves content
// to the right, i.e. out(u) = in(u - dx). Operates on a real array in place-safe
// fashion and returns a new array.
export function shiftPadded(img, w, h, dx, dy) {
  const f = new FFT2(w, h);
  f.re.set(img);
  f.forward();
  const { fx, fy } = frequencies(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const ang = -2 * Math.PI * (fx[x] * dx / w + fy[y] * dy / h);
      const c = Math.cos(ang), s = Math.sin(ang);
      const a = f.re[i], b = f.im[i];
      f.re[i] = a * c - b * s;
      f.im[i] = a * s + b * c;
    }
  }
  f.inverse();
  return f.re;
}

// Complex Fourier multiplier exp(-i 2pi (fx dx / w + fy dy / h)).
export function phaseMultiplier(w, h, dx, dy) {
  const re = new Float64Array(w * h);
  const im = new Float64Array(w * h);
  const { fx, fy } = frequencies(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ang = -2 * Math.PI * (fx[x] * dx / w + fy[y] * dy / h);
      re[y * w + x] = Math.cos(ang);
      im[y * w + x] = Math.sin(ang);
    }
  }
  return { re, im };
}

export function complexMulInto(outRe, outIm, ar, ai, br, bi) {
  for (let i = 0; i < outRe.length; i++) {
    outRe[i] = ar[i] * br[i] - ai[i] * bi[i];
    outIm[i] = ar[i] * bi[i] + ai[i] * br[i];
  }
}
