// End-to-end inversion pipeline.
//
//   observations + user assumptions  ->  motion analysis -> operators
//   ->  regularised CG solve  ->  estimate + diagnostics
//
// The truth (latent / clean frames) is not an input here. Evaluation lives in
// evaluate.js and is given truth explicitly.

import { GEOM } from './synth.js';
import { buildOperators, makeParametricPath } from './synth.js';
import { Solver, dataSupportMap, reblurResidual, localFrequencySupport, priorContrast } from './solve.js';
import { registerSequence, candidateFromRegistration } from './motion.js';
import { clamp } from './util.js';

export function analyzeClip(observations, opts = {}) {
  const { obsW, obsH } = GEOM;
  const reg = registerSequence(observations, { refIndex: opts.refIndex, w: obsW, h: obsH });
  const candidate = candidateFromRegistration(reg, opts.rho ?? 0.8);
  return { reg, candidate };
}

function maskFor(observations, obsW, obsH, edge = 3) {
  const mask = new Float64Array(obsW * obsH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) {
      const t = Math.min(x, y, obsW - 1 - x, obsH - 1 - y);
      mask[y * obsW + x] = t >= edge ? 1 : clamp((t + 0.5) / (edge + 0.5));
    }
  }
  return observations.map(() => mask);
}

function solveWith(observations, built, assumption, opts = {}) {
  const { obsW, obsH, padW, padH, ox, oy } = GEOM;
  const frames = observations.length;
  const lambda = assumption.lambda ?? 0.02;
  const masks = maskFor(observations, obsW, obsH);
  const solve = Solver.create({
    observations, operators: built.operators, masks, padW, padH, obsW, obsH, ox, oy,
    lambda, tv: assumption.tv ?? 0, regScale: assumption.regScale ?? 0.6,
    maxIter: assumption.maxIter ?? 50, tol: assumption.tol ?? 2e-5,
  });
  const est = new Float64Array(obsW * obsH);
  for (let y = 0; y < obsH; y++) {
    for (let x = 0; x < obsW; x++) est[y * obsW + x] = solve.x[(y + oy) * padW + (x + ox)];
  }
  return { solve, est };
}

export function buildAssumption(assumption, frames) {
  const refIndex = assumption.refIndex ?? Math.floor(frames / 2);
  const rho = assumption.rho ?? 0.8;
  const pathFn = makeParametricPath({
    kind: assumption.kind ?? 'linear',
    vx: assumption.vx ?? 0, vy: assumption.vy ?? 0,
    amp: assumption.amp ?? 0, freq: assumption.freq ?? 0,
    omega: assumption.omega ?? 0, pivotX: assumption.pivotX ?? 0, pivotY: assumption.pivotY ?? 0,
  });
  const speed = Math.hypot(assumption.vx ?? 0, assumption.vy ?? 0);
  const radius = Math.max(6, 12 + speed * rho * 1.2);
  const built = buildOperators(pathFn, frames, refIndex, rho, assumption.exposureKind ?? 'uniform', radius);
  return { pathFn, built, refIndex, rho };
}

// assumption: { kind, vx, vy, rho, exposureKind, refIndex, lambda, tv, maxIter }
export function invert(observations, assumption, opts = {}) {
  const { obsW, obsH, padW, padH, ox, oy } = GEOM;
  const frames = observations.length;
  const { built, refIndex, rho } = buildAssumption(assumption, frames);
  const lambda = assumption.lambda ?? 0.02;
  const { solve, est } = solveWith(observations, built, { ...assumption, lambda }, opts);

  const check = reblurResidual(est, observations, built.operators, padW, padH, obsW, obsH, ox, oy);
  let support = null;
  let contrast = null;
  let altEstimate = null;
  if (opts.computeSupport !== false) {
    support = localFrequencySupport(
      observations, built.operators, padW, padH, obsW, obsH, ox, oy,
      { lambda, regScale: assumption.regScale ?? 0.6, estimate: est },
    );
    // A second solve with a much stronger prior; the difference shows structure
    // that only exists because the prior allows it.
    const alt = solveWith(observations, built, { ...assumption, lambda: lambda * 40, maxIter: Math.min(assumption.maxIter ?? 50, 25) }, opts);
    altEstimate = alt.est;
    contrast = priorContrast(est, alt.est, obsW, obsH);
  }

  return {
    id: assumption.id ?? null,
    label: assumption.label ?? null,
    estimate: Float64Array.from(est),
    iterations: solve.iterations,
    history: solve.history,
    finalRelResidual: solve.finalRelResidual,
    kernel: built.kernels[refIndex],
    kernels: built.kernels,
    displacements: built.displacements,
    support,
    contrast,
    altEstimate,
    reblur: check,
    operators: built.operators,
    assumptions: {
      kind: assumption.kind ?? 'linear', vx: assumption.vx ?? 0, vy: assumption.vy ?? 0,
      amp: assumption.amp ?? 0, freq: assumption.freq ?? 0,
      omega: assumption.omega ?? 0, pivotX: assumption.pivotX ?? 0, pivotY: assumption.pivotY ?? 0,
      rho, exposureKind: assumption.exposureKind ?? 'uniform', refIndex,
      lambda, tv: assumption.tv ?? 0, regScale: assumption.regScale ?? 0.6,
      maxIter: assumption.maxIter ?? 50,
    },
  };
}

export function residualStats(reblur, obsW, obsH, margin = 4) {
  const perFrame = [];
  for (const r of reblur) {
    let se = 0, sae = 0, n = 0, peak = 0;
    for (let y = margin; y < obsH - margin; y++) {
      for (let x = margin; x < obsW - margin; x++) {
        const v = r.residual[y * obsW + x];
        se += v * v; sae += Math.abs(v); n++;
        peak = Math.max(peak, Math.abs(v));
      }
    }
    perFrame.push({ rmse: Math.sqrt(se / n), mae: sae / n, peak });
  }
  let se = 0, n = 0, peak = 0;
  for (const r of reblur) {
    for (let y = margin; y < obsH - margin; y++) {
      for (let x = margin; x < obsW - margin; x++) {
        const v = r.residual[y * obsW + x];
        se += v * v; n++;
        peak = Math.max(peak, Math.abs(v));
      }
    }
  }
  return { perFrame, rmse: Math.sqrt(se / n), peak, margin };
}

// Run several motion hypotheses and rank them by the forward residual. The
// key behaviour: if two kernels explain the observations equally well, we keep
// both and report the ambiguity instead of silently choosing one.
export function compareHypotheses(observations, hypotheses, opts = {}) {
  const { obsW, obsH } = GEOM;
  const results = [];
  for (const h of hypotheses) {
    const r = invert(observations, h, { computeSupport: false });
    const st = residualStats(r.reblur, obsW, obsH);
    results.push({ ...r, forwardRmse: st.rmse, forwardPeak: st.peak, residual: st });
  }
  const scored = results
    .map((r) => ({ id: r.id, label: r.label, forwardRmse: r.forwardRmse, result: r }))
    .sort((a, b) => a.forwardRmse - b.forwardRmse);
  const best = scored[0];
  const second = scored[1] ?? null;
  const margin = second
    ? (second.forwardRmse - best.forwardRmse) / Math.max(best.forwardRmse, 1e-6)
    : 1;
  return {
    results,
    scored,
    bestId: best.id,
    margin,
    // Two hypotheses whose forward residuals differ by less than 5% are treated
    // as indistinguishable by the data. This threshold is stated in the UI.
    ambiguous: Boolean(second) && margin < 0.05,
  };
}

// Local-nearby hypothesis scan: does the data care about a small change in the
// assumed velocity? Used to report a resolution limit ("ambiguity radius").
export function velocityAmbiguity(observations, assumption, opts = {}) {
  const base = compareHypotheses(
    observations,
    [{ ...assumption, id: 'base', label: 'assumed' }],
    opts,
  );
  const baseRmse = base.scored[0].forwardRmse;
  const deltas = opts.deltas ?? [0.25, 0.5, 1.0, 2.0];
  const rows = [];
  const angle = Math.atan2(assumption.vy ?? 0, assumption.vx ?? 0);
  const speed = Math.hypot(assumption.vx ?? 0, assumption.vy ?? 0);
  for (const dv of deltas) {
    for (const sign of [-1, 1]) {
      const s2 = Math.max(0, speed + sign * dv);
      const h = {
        ...assumption,
        id: `${sign > 0 ? 'plus' : 'minus'}${dv}`,
        label: `speed ${sign > 0 ? '+' : '-'}${dv}`,
        vx: s2 * Math.cos(angle),
        vy: s2 * Math.sin(angle),
      };
      const r = invert(observations, h, { computeSupport: false });
      const st = residualStats(r.reblur, GEOM.obsW, GEOM.obsH);
      rows.push({ delta: sign * dv, forwardRmse: st.rmse, estimate: r.estimate });
    }
  }
  return { baseRmse, rows, threshold: baseRmse * 1.05 };
}

export { dataSupportMap };

// Explore the two highest-likelihood *explanations* of the same blurred input:
// a constant-velocity kernel and its reverse, plus a stationary alternative.
// A symmetric exposure kernel (uniform shutter) and a locally linear image make
// h and h(reversed) nearly identical, so the data often cannot pick one; the
// returned `ambiguous` flag says so instead of silently choosing. Direction is
// resolved when inter-frame registration breaks the tie.
export function exploreMotionAmbiguity(observations, assumption, opts = {}) {
  const angle = Math.atan2(assumption.vy ?? 0, assumption.vx ?? 0);
  const speed = Math.hypot(assumption.vx ?? 0, assumption.vy ?? 0);
  const alt = Math.max(speed * 0.75, 1.5);
  const base = {
    ...assumption,
    kind: assumption.kind ?? 'linear',
    exposureKind: assumption.exposureKind ?? 'uniform',
  };
  const candidates = [
    { ...base, id: 'forward', label: 'forward velocity' },
    { ...base, id: 'reverse', label: 'reversed velocity', vx: -base.vx, vy: -base.vy },
    { ...base, id: 'slower', label: 'slower path (0.75x)', vx: alt * Math.cos(angle), vy: alt * Math.sin(angle) },
    { ...base, id: 'stationary', label: 'stationary', vx: 0, vy: 0 },
  ];
  const cmp = compareHypotheses(observations, candidates, opts);
  // Register the sequence to see whether inter-frame motion resolves direction.
  const reg = registerSequence(observations, {
    refIndex: assumption.refIndex, w: GEOM.obsW, h: GEOM.obsH,
  });
  const cand = candidateFromRegistration(reg, assumption.rho ?? 0.8);
  const regAngle = Math.atan2(cand.vy, cand.vx);
  const fwdAgree = Math.cos(regAngle - angle) > 0.3;
  return {
    ...cmp,
    registration: cand,
    registrationAgreesForward: speed < 0.6 ? null : fwdAgree,
    // When the data cannot separate the hypotheses, both are returned as valid
    // explanations; the UI must show both.
    kept: cmp.ambiguous
      ? cmp.scored.slice(0, 2).map((s) => s.id)
      : [cmp.bestId],
    lambdaUsed: assumption.lambda ?? 0.02,
  };
}
