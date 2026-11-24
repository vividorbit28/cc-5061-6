// Conversion of kernel arrays and maps to browser-friendly JSON.

export function kernelsForUI(kernel) {
  if (!kernel) return null;
  const { data, size, radius } = kernel;
  const arr = Array.from(data, (v) => Math.round(v * 1e6) / 1e6);
  return { size, radius, data: arr };
}

export function kernelToImage(kernel, scale = 1) {
  // Render a kernel heatmap into an RGBA-ish RGB buffer for display.
  const { data, size } = kernel;
  let max = 0;
  for (const v of data) max = Math.max(max, v);
  const out = new Uint8Array(size * size * 3);
  for (let i = 0; i < size * size; i++) {
    const t = max > 0 ? Math.pow(data[i] / max, 0.6) : 0;
    // simple blue->white ramp
    out[3 * i] = Math.round(255 * Math.min(1, t * 1.4));
    out[3 * i + 1] = Math.round(255 * Math.min(1, t * 1.1));
    out[3 * i + 2] = Math.round(255 * Math.min(1, 0.35 + t * 0.65));
  }
  return out;
}

export function upscaleGray(img, w, h, factor) {
  const ow = w * factor;
  const oh = h * factor;
  const out = new Float64Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) out[y * ow + x] = img[((y / factor) | 0) * w + ((x / factor) | 0)];
  }
  return out;
}
