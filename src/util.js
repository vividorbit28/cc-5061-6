// Small numeric helpers shared by the generator, kernel, and inversion.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hash2(ix, iy, seed) {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(seed, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

export function clamp(x, lo = 0, hi = 1) {
  return x < lo ? lo : x > hi ? hi : x;
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function smoothstep(t) {
  t = clamp(t);
  return t * t * (3 - 2 * t);
}

export function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

export function rms(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s / a.length);
}

export function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

export function median(arr) {
  const a = Float64Array.from(arr).sort();
  const n = a.length;
  if (!n) return 0;
  return n % 2 ? a[(n - 1) >> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

export function robustSigma(arr) {
  const m = median(arr);
  const dev = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) dev[i] = Math.abs(arr[i] - m);
  return 1.4826 * median(dev);
}

export function percentile(arr, p) {
  const a = Float64Array.from(arr).sort();
  const idx = clamp(Math.round((p / 100) * (a.length - 1)), 0, a.length - 1);
  return a[idx];
}

// Bilinear scene sampler with clamped edges. Coordinates are pixel centres.
export function bilinear(img, w, h, x, y) {
  const fx = clamp(x, 0, w - 1);
  const fy = clamp(y, 0, h - 1);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const a = img[y0 * w + x0];
  const b = img[y0 * w + x1];
  const c = img[y1 * w + x0];
  const d = img[y1 * w + x1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

export function cropSample(img, w, h, cx, cy, outW, outH) {
  const out = new Float64Array(outW * outH);
  const x0 = cx - outW / 2;
  const y0 = cy - outH / 2;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      out[y * outW + x] = bilinear(img, w, h, x0 + x, y0 + y);
    }
  }
  return out;
}

// Catmull-Rom bicubic sampling on a clamped image.
function cubic(p0, p1, p2, p3, t) {
  return (
    p1 +
    0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))
  );
}

export function bicubic(img, w, h, x, y) {
  const fx = clamp(x, 0, w - 1);
  const fy = clamp(y, 0, h - 1);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const out = new Float64Array(4);
  for (let j = 0; j < 4; j++) {
    const yy = clamp(y0 - 1 + j, 0, h - 1);
    const p0 = img[yy * w + clamp(x0 - 1, 0, w - 1)];
    const p1 = img[yy * w + clamp(x0, 0, w - 1)];
    const p2 = img[yy * w + clamp(x0 + 1, 0, w - 1)];
    const p3 = img[yy * w + clamp(x0 + 2, 0, w - 1)];
    out[j] = cubic(p0, p1, p2, p3, tx);
  }
  return cubic(out[0], out[1], out[2], out[3], ty);
}

// Resample with an arbitrary translation (bilinear; used for reporting only).
export function translateImage(img, w, h, dx, dy) {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) out[y * w + x] = bilinear(img, w, h, x - dx, y - dy);
  }
  return out;
}

export function gaussianKernel1d(sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float64Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    s += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return { k, r };
}

export function blurSeparable(img, w, h, sigma) {
  if (sigma <= 0) return Float64Array.from(img);
  const { k, r } = gaussianKernel1d(sigma);
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += k[j + r] * img[y * w + clamp(x + j, 0, w - 1)];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += k[j + r] * tmp[clamp(y + j, 0, h - 1) * w + x];
      out[y * w + x] = s;
    }
  }
  return out;
}

export function psnr(a, b, peak = 1) {
  let se = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    se += d * d;
  }
  const mse = se / a.length;
  if (mse <= 1e-15) return 99;
  return 10 * Math.log10((peak * peak) / mse);
}

// Standard 11x11 Gaussian-windowed SSIM on grayscale [0,1].
export function ssim(a, b, w, h) {
  const C1 = 0.01 * 0.01;
  const C2 = 0.03 * 0.03;
  const sigma = 1.5;
  const { k, r } = gaussianKernel1d(sigma);
  const mua = new Float64Array(w * h);
  const mub = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sa = 0, sb = 0;
      for (let j = -r; j <= r; j++) {
        const yy = clamp(y + j, 0, h - 1) * w;
        for (let i = -r; i <= r; i++) {
          const ww = k[i + r] * k[j + r];
          const xx = clamp(x + i, 0, w - 1);
          sa += ww * a[yy + xx];
          sb += ww * b[yy + xx];
        }
      }
      mua[y * w + x] = sa;
      mub[y * w + x] = sb;
    }
  }
  let total = 0;
  let count = 0;
  const step = 2;
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      let saa = 0, sbb = 0, sab = 0;
      for (let j = -r; j <= r; j++) {
        const yy = clamp(y + j, 0, h - 1) * w;
        for (let i = -r; i <= r; i++) {
          const ww = k[i + r] * k[j + r];
          const xx = clamp(x + i, 0, w - 1);
          const da = a[yy + xx] - mua[y * w + x];
          const db = b[yy + xx] - mub[y * w + x];
          saa += ww * da * da;
          sbb += ww * db * db;
          sab += ww * da * db;
        }
      }
      const num = (2 * mua[y * w + x] * mub[y * w + x] + C1) * (2 * sab + C2);
      const den = (mua[y * w + x] ** 2 + mub[y * w + x] ** 2 + C1) * (saa + sbb + C2);
      total += num / den;
      count++;
    }
  }
  return total / count;
}

// Interior mask helper: true away from the unreliable convolution border.
export function interiorIndices(w, h, margin) {
  const idx = [];
  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) idx.push(y * w + x);
  }
  return Int32Array.from(idx);
}

export function metricsOn(a, b, idx) {
  let se = 0;
  let sa = 0;
  let sb = 0;
  for (const i of idx) {
    const d = a[i] - b[i];
    se += d * d;
    sa += a[i];
    sb += b[i];
  }
  const n = idx.length;
  return { mse: se / n, mae: 0, meanA: sa / n, meanB: sb / n };
}

export function angleDiffDeg(a, b) {
  let d = Math.abs(a - b) % 180;
  if (d > 90) d = 180 - d;
  return d;
}

export function fmt(x, digits = 3) {
  if (!Number.isFinite(x)) return 'n/a';
  return Number(x).toFixed(digits);
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// Turbo-like colormap for residual / reliability maps (polynomial approximation).
export function turbo(t) {
  t = clamp(t);
  const r =
    0.13572138 + 4.6153926 * t - 42.66032258 * t * t + 132.13108234 * t ** 3 -
    152.94239396 * t ** 4 + 59.28637943 * t ** 5;
  const g =
    0.09140261 + 2.19418839 * t + 4.84296658 * t * t - 14.18503333 * t ** 3 +
    4.27729857 * t ** 4 + 2.82956604 * t ** 5;
  const b =
    0.1066733 + 12.64194608 * t - 60.58204836 * t * t + 110.36276771 * t ** 3 -
    89.90310912 * t ** 4 + 27.34824973 * t ** 5;
  return [clamp(r), clamp(g), clamp(b)];
}

export function to8(x) {
  return Math.max(0, Math.min(255, Math.round(x * 255)));
}

export function grayToRGB(img) {
  const out = new Uint8Array(img.length * 3);
  for (let i = 0; i < img.length; i++) {
    const v = to8(img[i]);
    out[3 * i] = v;
    out[3 * i + 1] = v;
    out[3 * i + 2] = v;
  }
  return out;
}

export function heatToRGB(img, maxVal, mask) {
  const out = new Uint8Array(img.length * 3);
  const inv = maxVal > 0 ? 1 / maxVal : 0;
  for (let i = 0; i < img.length; i++) {
    if (mask && !mask[i]) {
      out[3 * i] = 20;
      out[3 * i + 1] = 20;
      out[3 * i + 2] = 24;
      continue;
    }
    const [r, g, b] = turbo(clamp(img[i] * inv));
    out[3 * i] = to8(r);
    out[3 * i + 1] = to8(g);
    out[3 * i + 2] = to8(b);
  }
  return out;
}

// Mix a grayscale base with a heatmap overlay (used for prior-reliance maps).
export function overlayHeatRGB(base, heat, maxVal, alpha = 0.55, mask) {
  const out = new Uint8Array(base.length * 3);
  const inv = maxVal > 0 ? 1 / maxVal : 0;
  for (let i = 0; i < base.length; i++) {
    const v = to8(base[i]);
    if (mask && !mask[i]) {
      out[3 * i] = v;
      out[3 * i + 1] = v;
      out[3 * i + 2] = v;
      continue;
    }
    const [r, g, b] = turbo(clamp(heat[i] * inv));
    out[3 * i] = Math.round(v * (1 - alpha) + 255 * r * alpha);
    out[3 * i + 1] = Math.round(v * (1 - alpha) + 255 * g * alpha);
    out[3 * i + 2] = Math.round(v * (1 - alpha) + 255 * b * alpha);
  }
  return out;
}
