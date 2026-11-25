#!/usr/bin/env node
// Auxiliary command-line runner for the layered (multi-motion) blur inversion.
// The graphical player (npm start) is the primary entry point; this CLI mirrors
// the same src/layers.js kernel for batch checks and offline forward verification.
//
// Examples:
//   node scripts/layered-cli.mjs --case opposed
//   node scripts/layered-cli.mjs --case static-bg --compare-strategies
//   node scripts/layered-cli.mjs --case opposed --frame-start 1 --frame-end 3
//   node scripts/layered-cli.mjs --case opposed --alternations 1 --png-dir out/layered
//
// Truth masks/velocities are used ONLY for the optional scoring block, never as
// solver input (coverage is data-driven unless --explicit-masks is given).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { GEOM } from '../src/synth.js';
import {
  makeLayeredClip, invertLayered, invertSingleLayer,
  compareLayeredStrategies, layeredResidualRmse,
} from '../src/layers.js';
import { psnr } from '../src/util.js';
import { encodePNG } from '../server/media.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const CASES = {
  opposed: {
    title: 'opposed fg/bg motion (frozen-boundary model)',
    opts: { seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen', bgMotion: { type: 'pan', params: { vx: 4.0, vy: 0.6 } }, fgMotion: { vx: -5.0, vy: -1.6 } },
  },
  'static-bg': {
    title: 'moving foreground over a static background',
    opts: { seed: 31, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'frozen', bgMotion: { type: 'pan', params: { vx: 0, vy: 0 } }, fgMotion: { vx: -4.5, vy: 1.2 } },
  },
  exact: {
    title: 'exact time-sampled boundaries (frozen-approximation stress test)',
    opts: { seed: 21, frames: 5, refIndex: 2, rho: 0.8, noiseSigma: 0.002, quantize: true, boundaryMode: 'exact', bgMotion: { type: 'pan', params: { vx: 4.0, vy: 0.6 } }, fgMotion: { vx: -5.0, vy: -1.6 } },
  },
};

function parseArgs(argv) {
  const a = { case: 'opposed', lambda: 0.01, maxIter: 35, alternations: 0, compareStrategies: false, explicitMasks: false, frameStart: null, frameEnd: null, pngDir: null };
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i];
    const next = () => argv[++i];
    if (t === '--case') a.case = next();
    else if (t === '--lambda') a.lambda = Number(next());
    else if (t === '--max-iter') a.maxIter = Number(next());
    else if (t === '--alternations') a.alternations = Number(next());
    else if (t === '--frame-start') a.frameStart = Number(next());
    else if (t === '--frame-end') a.frameEnd = Number(next());
    else if (t === '--png-dir') a.pngDir = next();
    else if (t === '--compare-strategies') a.compareStrategies = true;
    else if (t === '--explicit-masks') a.explicitMasks = true;
    else if (t === '--help' || t === '-h') { printHelp(); process.exit(0); }
    else throw new Error(`unknown argument ${t}`);
  }
  return a;
}

function printHelp() {
  console.log(`layered blur inversion CLI
  --case <opposed|static-bg|exact>   synthetic scenario (default opposed)
  --lambda <n>                       Tikhonov weight (default 0.01)
  --max-iter <n>                     CG iterations (default 35)
  --alternations <0|1>               protected per-layer velocity re-estimation
  --frame-start <i> --frame-end <i>  blur segment (default whole clip)
  --compare-strategies               independent vs shared-background, pick by residual
  --explicit-masks                   use truth masks as lock regions (oracle; evaluation)
  --png-dir <dir>                    write estimate/mask/uncertainty PNGs
  -h, --help                         this help`);
}

const { obsW, obsH, padW, padH, ox, oy } = GEOM;
const embed = (m) => {
  const c = new Float64Array(padW * padH);
  for (let y = 0; y < obsH; y++) for (let x = 0; x < obsW; x++) c[(y + oy) * padW + (x + ox)] = m[y * obsW + x];
  return c;
};

function writePng(dir, name, img) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, name), encodePNG(img, obsW, obsH));
}

const args = parseArgs(process.argv);
const entry = CASES[args.case];
if (!entry) throw new Error(`unknown case ${args.case}; choose one of ${Object.keys(CASES).join(', ')}`);

const fullClip = makeLayeredClip(entry.opts);
const total = fullClip.obs.length;
const start = args.frameStart ?? 0;
const end = args.frameEnd ?? total - 1;
const obs = fullClip.obs.slice(start, end + 1);
const refIndex = Math.min(obs.length - 1, Math.max(0, fullClip.refIndex - start));
const bg = fullClip.truthVelocities.bg;
const fg = fullClip.truthVelocities.fg;

const layers = [
  { kind: 'linear', vx: bg.vx, vy: bg.vy },
  { kind: 'linear', vx: fg.vx, vy: fg.vy },
];
// Oracle path pins the truth coverage to demonstrate the closed loop; the
// default (data-driven) path derives coverage from the observations.
if (args.explicitMasks) {
  layers[0].maskCanvas = embed(fullClip.truthMasks.bg); layers[0].fixedCoverage = true;
  layers[1].maskCanvas = embed(fullClip.truthMasks.fg);
}
const spec = {
  layers, refIndex, rho: entry.opts.rho, exposureKind: entry.opts.exposureKind ?? 'uniform',
  lambda: args.lambda, maxIter: args.maxIter, alternations: args.alternations,
  estimateMasks: !args.explicitMasks,
  bgVelocity: { x: bg.vx, y: bg.vy },
};

console.log(`case: ${args.case} — ${entry.title}`);
console.log(`segment frames ${start}..${end} (local ref ${refIndex}) · masks: ${args.explicitMasks ? 'explicit truth (oracle)' : 'data-driven'}`);

let main, strategy = null;
if (args.compareStrategies) {
  const cmp = compareLayeredStrategies(obs, spec);
  main = cmp.results[cmp.selected];
  strategy = cmp;
  console.log('strategy scores (forward RMSE):',
    `independent ${cmp.scores.independent.toFixed(5)}`,
    `shared-bg ${cmp.scores['shared-bg'].toFixed(5)}`,
    `→ selected ${cmp.selected} (margin ${(cmp.margin * 100).toFixed(1)}%)`);
} else {
  main = invertLayered(obs, spec);
}

const layeredRmse = layeredResidualRmse(main, obs);
const single = invertSingleLayer(obs, { vx: bg.vx, vy: bg.vy, refIndex, rho: spec.rho, exposureKind: spec.exposureKind, lambda: args.lambda, maxIter: args.maxIter });
const singleRmse = layeredResidualRmse(single, obs);
const truthRef = fullClip.clean[start + refIndex];

console.log('--- forward verification (closed-loop re-blur) ---');
console.log(`layered forward RMSE : ${layeredRmse.toFixed(5)}`);
console.log(`single-motion RMSE   : ${singleRmse.toFixed(5)}`);
console.log(`residual change      : ${(((singleRmse - layeredRmse) / singleRmse) * 100).toFixed(1)}% vs single-motion`);
console.log(`composite PSNR       : ${psnr(main.estimate, truthRef).toFixed(2)} dB`);
console.log(`single-motion PSNR   : ${psnr(single.estimate, truthRef).toFixed(2)} dB`);
console.log(`layers ${main.layers.length} · CG iters ${main.iterations} · per-layer kernels ${main.kernels.map((k) => `${k.size}x${k.size}`).join(', ')}`);
main.layers.forEach((l, i) => console.log(`  layer ${i} velocity (${(l.vx ?? 0).toFixed(2)}, ${(l.vy ?? 0).toFixed(2)})`));
if (strategy) console.log(`strategy field on result: ${main.strategy}`);

if (args.pngDir) {
  const dir = path.isAbsolute(args.pngDir) ? args.pngDir : path.join(ROOT, args.pngDir);
  writePng(dir, `${args.case}-estimate.png`, main.estimate);
  writePng(dir, `${args.case}-single.png`, single.estimate);
  writePng(dir, `${args.case}-residual.png`, main.reblur[refIndex].residual.map((v) => 0.5 + 4 * v));
  main.layerEstimates.forEach((a, i) => writePng(dir, `${args.case}-layer${i}.png`, a));
  main.layerMasks.forEach((a, i) => writePng(dir, `${args.case}-mask${i}.png`, a));
  main.uncertainty.forEach((a, i) => writePng(dir, `${args.case}-uncertainty${i}.png`, a));
  console.log(`PNGs written to ${dir}`);
}
