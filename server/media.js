// PNG / MP4 encoding helpers. PNG is written by hand (no deps); MP4 is produced
// with ffmpeg-static when available, with a PPM sequence fallback.

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// PNG (8-bit grayscale or RGB), with filter type 0 rows.
// ---------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

function pngHeader(w, h, colorType) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = colorType; // 0 gray, 2 RGB
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  return chunk('IHDR', ihdr);
}

export function encodePNG(gray, w, h) {
  const raw = Buffer.alloc((w + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const v = Math.max(0, Math.min(1, gray[y * w + x]));
      raw[y * (w + 1) + 1 + x] = Math.round(v * 255);
    }
  }
  const idat = zlib.deflateSync(raw, { level: 6 });
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngHeader(w, h, 0),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

export function encodePNGColor(rgb, w, h) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3;
      raw[y * (w * 3 + 1) + 1 + x * 3] = rgb[i];
      raw[y * (w * 3 + 1) + 2 + x * 3] = rgb[i + 1];
      raw[y * (w * 3 + 1) + 3 + x * 3] = rgb[i + 2];
    }
  }
  const idat = zlib.deflateSync(raw, { level: 6 });
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngHeader(w, h, 2),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

export function encodeRawGray(gray, w, h) {
  const buf = Buffer.alloc(w * h);
  for (let i = 0; i < gray.length; i++) buf[i] = Math.round(Math.max(0, Math.min(1, gray[i])) * 255);
  return buf;
}

// Overlay a simple text banner into a grayscale frame (5x7 bitmap font).
const FONT = {
  ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
  A: ['02220', '20012', '20012', '22222', '20012', '20012', '20012'],
  B: ['22220', '20012', '20012', '22220', '20012', '20012', '22220'],
  C: ['02222', '20000', '20000', '20000', '20000', '20000', '02222'],
  D: ['22220', '20012', '20012', '20012', '20012', '20012', '22220'],
  E: ['22222', '20000', '20000', '22220', '20000', '20000', '22222'],
  F: ['22222', '20000', '20000', '22220', '20000', '20000', '20000'],
  G: ['02220', '20002', '20000', '20222', '20002', '20002', '02220'],
  H: ['20012', '20012', '20012', '22222', '20012', '20012', '20012'],
  I: ['22222', '02000', '02000', '02000', '02000', '02000', '22222'],
  J: ['22222', '00020', '00020', '00020', '20020', '20020', '02200'],
  K: ['20012', '20020', '20200', '22000', '20200', '20020', '20012'],
  L: ['20000', '20000', '20000', '20000', '20000', '20000', '22222'],
  M: ['21112', '22222', '21212', '20202', '20002', '20002', '20002'],
  N: ['20012', '22012', '22212', '20222', '20022', '20012', '20012'],
  O: ['02220', '20012', '20012', '20012', '20012', '20012', '02220'],
  P: ['22220', '20012', '20012', '22220', '20000', '20000', '20000'],
  Q: ['02220', '20012', '20012', '20012', '20212', '20020', '02202'],
  R: ['22220', '20012', '20012', '22220', '20200', '20020', '20012'],
  S: ['02222', '20000', '20000', '02220', '00002', '00002', '22220'],
  T: ['22222', '02000', '02000', '02000', '02000', '02000', '02000'],
  U: ['20012', '20012', '20012', '20012', '20012', '20012', '02220'],
  V: ['20012', '20012', '20012', '20012', '20012', '02020', '00200'],
  W: ['20002', '20002', '20002', '20202', '21212', '22222', '21112'],
  X: ['20012', '20012', '02020', '00200', '02020', '20012', '20012'],
  Y: ['20012', '20012', '02020', '00200', '00200', '00200', '00200'],
  Z: ['22222', '00012', '00020', '00200', '02000', '20000', '22222'],
  '0': ['02220', '20012', '20022', '20202', '22002', '20012', '02220'],
  '1': ['02000', '12000', '02000', '02000', '02000', '02000', '22200'],
  '2': ['02220', '20012', '00012', '00020', '00200', '02000', '22222'],
  '3': ['22222', '00020', '00200', '00020', '00002', '20002', '02220'],
  '4': ['00020', '00220', '02020', '20020', '22222', '00020', '00020'],
  '5': ['22222', '20000', '22220', '00012', '00002', '20012', '02220'],
  '6': ['00220', '02000', '20000', '22220', '20012', '20012', '02220'],
  '7': ['22222', '00012', '00020', '00020', '00200', '00200', '00200'],
  '8': ['02220', '20012', '20012', '02220', '20012', '20012', '02220'],
  '9': ['02220', '20012', '20012', '02222', '00002', '00020', '02200'],
  '.': ['00000', '00000', '00000', '00000', '00000', '02000', '02000'],
  ',': ['00000', '00000', '00000', '00000', '00000', '02000', '20000'],
  ':': ['00000', '02000', '02000', '00000', '02000', '02000', '00000'],
  '-': ['00000', '00000', '00000', '22222', '00000', '00000', '00000'],
  '+': ['00000', '00200', '00200', '22222', '00200', '00200', '00000'],
  '%': ['20002', '20020', '00020', '00200', '02000', '20020', '20002'],
  '/': ['00002', '00020', '00020', '00200', '02000', '02000', '20000'],
  '(': ['00200', '02000', '20000', '20000', '20000', '02000', '00200'],
  ')': ['02000', '00200', '00020', '00020', '00020', '00200', '02000'],
  '=': ['00000', '00000', '22222', '00000', '22222', '00000', '00000'],
  '>': ['20000', '02000', '00200', '00020', '00200', '02000', '20000'],
  '<': ['00020', '00200', '02000', '20000', '02000', '00200', '00020'],
  '!': ['02000', '02000', '02000', '02000', '02000', '00000', '02000'],
  '_': ['00000', '00000', '00000', '00000', '00000', '00000', '22222'],
  '|': ['02000', '02000', '02000', '02000', '02000', '02000', '02000'],
  'x': ['00000', '00000', '20012', '02020', '00200', '02020', '20012'],
  'a': ['00000', '00000', '02220', '00012', '02222', '20012', '02222'],
  'b': ['20000', '20000', '22220', '20012', '20012', '20012', '22220'],
  'c': ['00000', '00000', '02222', '20000', '20000', '20000', '02222'],
  'd': ['00012', '00012', '02222', '20012', '20012', '20012', '02222'],
  'e': ['00000', '00000', '02220', '20012', '22222', '20000', '02220'],
  'f': ['00222', '02000', '22222', '02000', '02000', '02000', '02000'],
  'g': ['00000', '00000', '02222', '20012', '02222', '00012', '02220'],
  'h': ['20000', '20000', '22220', '20012', '20012', '20012', '20012'],
  'i': ['02000', '00000', '02000', '02000', '02000', '02000', '02000'],
  'j': ['00020', '00000', '00020', '00020', '00020', '20020', '02200'],
  'k': ['20000', '20000', '20020', '20200', '22000', '20200', '20020'],
  'l': ['02000', '02000', '02000', '02000', '02000', '02000', '00200'],
  'm': ['00000', '00000', '22220', '21212', '21212', '21212', '21212'],
  'n': ['00000', '00000', '22220', '20012', '20012', '20012', '20012'],
  'o': ['00000', '00000', '02220', '20012', '20012', '20012', '02220'],
  'p': ['00000', '00000', '22220', '20012', '22220', '20000', '20000'],
  'q': ['00000', '00000', '02222', '20012', '02222', '00012', '00012'],
  'r': ['00000', '00000', '20222', '22000', '20000', '20000', '20000'],
  's': ['00000', '00000', '02222', '20000', '02220', '00002', '22220'],
  't': ['02000', '02000', '22222', '02000', '02000', '02000', '00222'],
  'u': ['00000', '00000', '20012', '20012', '20012', '20012', '02222'],
  'v': ['00000', '00000', '20012', '20012', '20012', '02020', '00200'],
  'w': ['00000', '00000', '20002', '20002', '20202', '21212', '22222'],
  'y': ['00000', '00000', '20012', '20012', '02222', '00012', '02220'],
  'z': ['00000', '00000', '22222', '00020', '00200', '02000', '22222'],
  '*': ['00000', '02000', '22220', '02000', '00000', '00000', '00000'],
};

// Draw text into a float image. Returns nothing; mutates.
export function drawText(img, w, h, text, x0, y0, scale = 2, color = 1, bg = 0) {
  let cx = x0;
  for (const rawCh of text) {
    const ch = FONT[rawCh] ? rawCh : FONT[rawCh.toUpperCase()] ? rawCh.toUpperCase() : ' ';
    const glyph = FONT[ch];
    for (let gy = 0; gy < 7; gy++) {
      for (let gx = 0; gx < 5; gx++) {
        if (!glyph[gy] || glyph[gy][gx] === '0' || glyph[gy][gx] === undefined) continue;
        for (let sy = 0; sy < scale; sy++) {
          for (let sx = 0; sx < scale; sx++) {
            const px = cx + gx * scale + sx;
            const py = y0 + gy * scale + sy;
            if (px >= 0 && px < w && py >= 0 && py < h) img[py * w + px] = color;
          }
        }
      }
    }
    cx += 6 * scale;
    if (cx > w - 6 * scale) break;
  }
}

export function drawRect(img, w, h, x0, y0, x1, y1, color) {
  for (let y = Math.max(0, y0); y < Math.min(h, y1); y++) {
    for (let x = Math.max(0, x0); x < Math.min(w, x1); x++) img[y * w + x] = color;
  }
}

// ---------------------------------------------------------------------------
// ffmpeg discovery and MP4 encoding
// ---------------------------------------------------------------------------

let ffmpegPath;
export function findFFmpeg() {
  if (ffmpegPath !== undefined) return ffmpegPath;
  ffmpegPath = null;
  // 1) ffmpeg-static package (installed as a dependency)
  try {
    const mod = createRequire(import.meta.url)('ffmpeg-static');
    if (typeof mod === 'string' && fs.existsSync(mod)) ffmpegPath = mod;
  } catch {}
  // 2) PATH
  if (!ffmpegPath) {
    const which = spawnSync(process.platform === 'win32' ? 'where' : 'which', ['ffmpeg']);
    if (which.status === 0) {
      const p = which.stdout.toString().split('\n')[0].trim();
      if (p) ffmpegPath = p;
    }
  }
  return ffmpegPath;
}

import { createRequire } from 'node:module';

// Encode grayscale frames (array of Float64Array [0,1]) into an MP4 via ffmpeg.
// Frames are passed as raw gray8 on stdin. Optional per-frame banner text.
export async function encodeMP4(frames, w, h, outPath, opts = {}) {
  const ff = findFFmpeg();
  const fps = opts.fps ?? 12;
  if (!ff) {
    // Fallback: write PNG sequence plus a manifest. Playback still works in the
    // browser via the PNG sequence endpoint.
    const dir = outPath.replace(/\.mp4$/, '_frames');
    fs.mkdirSync(dir, { recursive: true });
    frames.forEach((f, i) => {
      fs.writeFileSync(path.join(dir, `f${String(i).padStart(4, '0')}.png`), encodePNG(f, w, h));
    });
    fs.writeFileSync(outPath.replace(/\.mp4$/, '.json'), JSON.stringify({ fallback: 'png-sequence', dir: path.basename(dir), w, h, fps }));
    return { ok: false, reason: 'ffmpeg-not-found', fallbackDir: dir };
  }
  const args = [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', `${w}x${h}`, '-r', String(fps),
    '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    outPath,
  ];
  await new Promise((resolve, reject) => {
    const p = spawn(ff, args, { stdio: ['pipe', 'ignore', 'pipe'] });
    let err = '';
    p.stderr.on('data', (d) => { err += d.toString(); });
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (${code}): ${err.slice(0, 500)}`));
    });
    let i = 0;
    const writeNext = () => {
      while (i < frames.length) {
        const buf = encodeRawGray(frames[i], w, h);
        i++;
        if (!p.stdin.write(buf)) {
          p.stdin.once('drain', writeNext);
          return;
        }
      }
      p.stdin.end();
    };
    writeNext();
  });
  return { ok: true, path: outPath };
}

export { path, os };
