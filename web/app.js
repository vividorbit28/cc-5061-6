// Blur Inversion Studio — front end.
// All numbers shown come from the server-side inversion kernel; this file only
// renders them and drives the player/controls. Display gain is cosmetic.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  catalog: null,
  clip: null,          // /api/clip response
  clipId: null,
  frame: 0,
  playing: false,
  cleanMode: false,
  jobId: null,
  result: null,
  truth: null,
  build: { running: false },
};

const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

init().catch((e) => {
  console.error(e);
  setEngine('failed');
  setJobStatus('failed', 'failed', String(e.message || e));
});

async function init() {
  setEngine('idle');
  const cat = await getJSON('/api/catalog');
  state.catalog = cat;
  renderClipList();
  $('#sbEngine').textContent = `engine: ${cat.geometry.obsW}×${cat.geometry.obsH} obs · padded FFT ${cat.geometry.padW}×${cat.geometry.padH}`;
  wireForm();
  wireTabs();
  wireResultControls();
  await selectClip(cat.clips[0].id);
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

function setEngine(s) {
  const el = $('#engineState');
  el.className = `pill ${s}`;
  el.textContent = s === 'running' ? 'solving…' : s === 'done' ? 'ready' : s === 'failed' ? 'error' : 'idle';
}

// ---------------------------------------------------------------------------
// Clip list
// ---------------------------------------------------------------------------

function renderClipList() {
  const list = $('#clipList');
  list.innerHTML = '';
  for (const c of state.catalog.clips) {
    const btn = document.createElement('button');
    btn.className = 'clip-item';
    btn.dataset.id = c.id;
    btn.innerHTML = `
      <canvas width="128" height="80"></canvas>
      <span>
        <span class="t">${c.title}</span>
        <span class="m">${c.spec.motionType} · ρ=${fmt(c.spec.rho, 2)} · σ=${fmt(c.spec.noiseSigma, 3)}</span>
      </span>
      <span class="tag">${c.spec.frames}f</span>`;
    btn.addEventListener('click', () => selectClip(c.id));
    list.appendChild(btn);
  }
}

async function selectClip(id) {
  stopPlay();
  state.clipId = id;
  $$('.clip-item').forEach((el) => el.classList.toggle('active', el.dataset.id === id));
  state.clip = await getJSON(`/api/clip?id=${encodeURIComponent(id)}`);
  state.frame = state.clip.refIndex;
  state.truth = null;
  state.result = null;
  clearResult();
  const spec = state.clip.spec;
  $('#metaClip').textContent = state.clip.title;
  $('#metaRho').textContent = fmt(spec.rho, 2);
  $('#metaExposure').textContent = spec.exposureKind;
  $('#metaNoise').textContent = fmt(spec.noiseSigma, 3);
  $('#frameSlider').max = String(state.clip.frames.length - 1);
  $('#frameSlider').value = String(state.frame);
  // Reference frame options
  const refSel = $('#refIndex');
  refSel.innerHTML = state.clip.frames.map((f, i) =>
    `<option value="${i}" ${i === state.clip.refIndex ? 'selected' : ''}>${i}${i === state.clip.refIndex ? ' (clip ref)' : ''}</option>`).join('');
  // Seed assumption from the clip spec's true motion params only as a
  // convenience default is NOT allowed; use the blind registration instead.
  applyRegistrationDefaults();
  drawRegistration();
  renderPlayerFrame();
  renderThumbnails();
  $('#resultHint').textContent = 'run an inversion to populate';
}

function applyRegistrationDefaults() {
  const c = state.clip.registration.candidate;
  const kind = state.clip.spec.motionType === 'shake' ? 'shake'
    : state.clip.spec.motionType === 'arc' ? 'arc'
      : state.clip.spec.motionType === 'stationary' ? 'stationary' : 'linear';
  $('#kind').value = kind;
  // The registered velocity is what the data alone suggests. The user sees this
  // and can override; it is intentionally not the generator's parameter.
  $('#vx').value = String(Math.round(c.vx * 10) / 10);
  $('#vy').value = String(Math.round(c.vy * 10) / 10);
  if (kind === 'arc') {
    $('#omega').value = '0.03';
  }
  $('#rho').value = String(state.clip.spec.rho);
  $('#exposureKind').value = state.clip.spec.exposureKind;
  syncSliders();
  toggleMotionFields();
}

function syncSliders() {
  const pairs = [
    ['rho', 'rhoOut', 2], ['vx', 'vxOut', 1], ['vy', 'vyOut', 1],
    ['amp', 'ampOut', 1], ['freq', 'freqOut', 2], ['omega', 'omegaOut', 3],
    ['tv', 'tvOut', 3], ['maxIter', 'maxIterOut', 0],
    ['dispGain', 'dispGainOut', 2],
  ];
  for (const [id, out, d] of pairs) {
    const el = document.getElementById(id);
    const o = document.getElementById(out);
    if (el && o) o.textContent = Number(el.value).toFixed(d);
  }
  const auto = $('#lambdaMode').value !== 'manual';
  $('#lambda').disabled = auto;
  const lo = document.getElementById('lambdaOut');
  if (lo) lo.textContent = auto ? 'auto' : Number($('#lambda').value).toFixed(3);
}

function toggleMotionFields() {
  const kind = $('#kind').value;
  $('#linearFields').classList.toggle('hidden', kind === 'arc');
  $('#shakeFields').classList.toggle('hidden', kind !== 'shake');
  $('#arcFields').classList.toggle('hidden', kind !== 'arc');
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

function renderPlayerFrame() {
  if (!state.clip) return;
  const f = state.clip.frames[state.frame];
  $('#playerFrame').src = state.cleanMode ? f.clean : f.obs;
  $('#frameIndexBadge').textContent = `frame ${state.frame}`;
  $('#frameSlider').value = String(state.frame);
}

function renderThumbnails() {
  // small previews on clip list canvases
  const items = $$('.clip-item');
  for (const el of items) {
    if (el.dataset.id !== state.clipId) continue;
    const cv = el.querySelector('canvas');
    const ctx = cv.getContext('2d');
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0, cv.width, cv.height);
    img.src = state.clip.frames[state.frame].obs;
  }
}

let playTimer = null;
function startPlay() {
  stopPlay();
  state.playing = true;
  $('#playToggle').textContent = '❚❚ pause';
  playTimer = setInterval(() => {
    state.frame = (state.frame + 1) % state.clip.frames.length;
    renderPlayerFrame();
  }, 380);
}
function stopPlay() {
  state.playing = false;
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  const b = $('#playToggle');
  if (b) b.textContent = '▶ play';
}

// ---------------------------------------------------------------------------
// Registration plot
// ---------------------------------------------------------------------------

function drawRegistration() {
  const cv = $('#regPlot');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!state.clip) return;
  const reg = state.clip.registration;
  const n = reg.perFrame.length;
  const cx = W / 2, cy = H / 2;
  ctx.strokeStyle = '#2a3244';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  // plot in a frame with origin at the sequence centroid to keep it on screen
  const maxAbs = Math.max(1, ...reg.perFrame.map((p) => Math.hypot(p.dx, p.dy)));
  const scale = Math.min((W - 40) / (2 * maxAbs), (H - 30) / (2 * maxAbs));
  ctx.strokeStyle = '#57b7ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  reg.perFrame.forEach((p, i) => {
    const x = cx + p.dx * scale;
    const y = cy + p.dy * scale;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  reg.perFrame.forEach((p, i) => {
    const x = cx + p.dx * scale;
    const y = cy + p.dy * scale;
    ctx.fillStyle = i === reg.refIndex ? '#7cf0c4' : '#57b7ff';
    ctx.beginPath(); ctx.arc(x, y, i === reg.refIndex ? 4 : 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#93a0b5';
    ctx.font = '9px monospace';
    ctx.fillText(String(i), x + 5, y - 4);
    if (p.spread > 1.5) {
      ctx.fillStyle = '#ffb45c';
      ctx.fillText('±' + p.spread.toFixed(1), x + 5, y + 8);
    }
  });
  const c = reg.candidate;
  $('#regCandidate').textContent = `vx ${fmt(c.vx, 2)} · vy ${fmt(c.vy, 2)} px/frame · |v| ${fmt(Math.hypot(c.vx, c.vy), 2)}`;
  $('#regSpread').textContent = `±${fmt(c.spread, 2)} px`;
}

// ---------------------------------------------------------------------------
// Form wiring
// ---------------------------------------------------------------------------

function wireForm() {
  const sync = () => syncSliders();
  $$('input[type=range]').forEach((el) => el.addEventListener('input', sync));
  $('#kind').addEventListener('change', toggleMotionFields);
  $('#lambdaMode').addEventListener('change', () => {
    const auto = $('#lambdaMode').value === 'auto';
    $('#lambda').disabled = auto;
    if (auto) {
      // reflect the server-side auto lambda if known, else keep last manual
    }
  });
  $('#exposureKind').addEventListener('change', () => {});
  $('#showClean').addEventListener('change', (e) => {
    state.cleanMode = e.target.checked;
    renderPlayerFrame();
    const badge = document.querySelector('.viewer-badges .badge.muted');
    if (badge) badge.textContent = state.cleanMode ? 'clean reference (truth, player only)' : 'blurred observation';
  });
  $('#btnFromReg').addEventListener('click', () => {
    const c = state.clip.registration.candidate;
    $('#vx').value = String(Math.round(c.vx * 10) / 10);
    $('#vy').value = String(Math.round(c.vy * 10) / 10);
    $('#kind').value = 'linear';
    toggleMotionFields();
    syncSliders();
  });
  $('#playToggle').addEventListener('click', () => (state.playing ? stopPlay() : startPlay()));
  $('#frameSlider').addEventListener('input', (e) => {
    stopPlay();
    state.frame = Number(e.target.value);
    renderPlayerFrame();
  });
  $('#assumptionForm').addEventListener('submit', (e) => {
    e.preventDefault();
    runInversion();
  });
  $('#btnBuild').addEventListener('click', () => runBuild());
  $('#dispGain').addEventListener('input', applyDisplayGain);
}

function currentAssumption() {
  const kind = $('#kind').value;
  const a = {
    kind,
    rho: Number($('#rho').value),
    exposureKind: $('#exposureKind').value,
    refIndex: Number($('#refIndex').value),
    tv: Number($('#tv').value),
    maxIter: Number($('#maxIter').value),
    lambdaMode: $('#lambdaMode').value,
  };
  if (kind === 'linear' || kind === 'stationary') {
    a.vx = kind === 'stationary' ? 0 : Number($('#vx').value);
    a.vy = kind === 'stationary' ? 0 : Number($('#vy').value);
  } else if (kind === 'shake') {
    a.vx = Number($('#vx').value);
    a.vy = Number($('#vy').value);
    a.amp = Number($('#amp').value);
    a.freq = Number($('#freq').value);
  } else if (kind === 'arc') {
    a.omega = Number($('#omega').value);
    a.pivotX = Number($('#pivotX').value);
    a.pivotY = Number($('#pivotY').value);
    a.vx = 0; a.vy = 0;
  }
  if (a.lambdaMode === 'manual') a.lambda = Number($('#lambda').value);
  return a;
}

// ---------------------------------------------------------------------------
// Inversion job
// ---------------------------------------------------------------------------

async function runInversion() {
  if (state.build.running) return;
  const body = { clipId: state.clipId, ...currentAssumption(), compare: $('#compare').checked };
  setEngine('running');
  setJobStatus('running', 'submitting', '');
  $('#btnInvert').disabled = true;
  const started = performance.now();
  try {
    const r = await fetch('/api/invert', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`invert → ${r.status}`);
    const { jobId } = await r.json();
    state.jobId = jobId;
    await pollJob(jobId, started);
  } catch (e) {
    setEngine('failed');
    setJobStatus('failed', 'error', String(e.message || e));
  } finally {
    $('#btnInvert').disabled = false;
  }
}

async function pollJob(jobId, started) {
  for (;;) {
    await sleep(320);
    const j = await getJSON(`/api/job?id=${jobId}`);
    setJobStatus(j.state, j.stage || j.state, j.progress != null ? `${Math.round(j.progress * 100)}%` : '');
    $('#sbJob').textContent = `job: ${j.state} ${j.stage || ''}`;
    if (j.state === 'failed') {
      setEngine('failed');
      setJobStatus('failed', 'failed', j.error || 'unknown error');
      showFailure(j.error || 'unknown error');
      return;
    }
    if (j.state === 'done') {
      const ms = Math.round(performance.now() - started);
      const res = j.result;
      state.result = res;
      renderResult(res, ms);
      setEngine('done');
      setJobStatus('done', 'done', `${ms} ms · ${res.iterations} CG iters`);
      return;
    }
  }
}

async function showFailure(error) {
  // Failure feeds back into the blur-inversion panel: show why and let the user
  // adjust assumptions rather than silently hiding it.
  $('#resultHint').textContent = 'inversion failed — see status';
  $('#jobDetail').textContent = 'see message';
  const el = $('#emptyResult');
  el.classList.remove('hidden');
  el.innerHTML = `<div class="empty-glyph">!</div><p><b>Inversion failed.</b></p><pre class="errpre">${escapeHTML(error)}</pre><p class="note">Reduce iterations, loosen λ, or check the assumed velocity. The failure is reported here and in the status bar.</p>`;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function setJobStatus(cls, stage, detail) {
  const box = $('#jobStatus');
  box.className = `job-status ${cls}`;
  $('#jobStage').textContent = stage;
  $('#jobDetail').textContent = detail || '';
  const bar = $('#jobBar');
  if (cls === 'running') bar.style.width = '35%';
  else if (cls === 'done') bar.style.width = '100%';
  else if (cls === 'failed') bar.style.width = '100%';
  else bar.style.width = '0%';
  if (cls === 'running') {
    // progress comes from polling; the bar is nudged by progress value below
  }
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------

function clearResult() {
  $('#resultWrap').classList.add('hidden');
  $('#emptyResult').classList.remove('hidden');
  $('#emptyResult').innerHTML = `<div class="empty-glyph">⇆</div>
    <p>Choose a clip, inspect its frames, set the motion/shutter assumption, then <b>Invert blur</b>.</p>
    <p class="note">The solver minimises a masked multi-frame data term plus a high-pass prior with conjugate gradients. It does not receive the clean frames.</p>`;
}

function renderResult(res, ms) {
  $('#emptyResult').classList.add('hidden');
  $('#resultWrap').classList.remove('hidden');
  $('#resultHint').textContent = `${res.iterations} CG iters · ${ms} ms · rel. residual ${fmt(res.finalRelResidual, 4)}`;
  $('#sbJob').textContent = `job: done (${res.iterations} it)`;
  $('#sbTime').textContent = `${ms} ms · ρ=${fmt(res.assumption.rho, 2)}`;
  const sl = document.getElementById('sbLambda');
  if (sl) sl.textContent = `λ=${fmt(res.assumption.lambda, 4)} (${res.autoLambda === res.assumption.lambda ? 'auto' : 'manual'}) · noise σ≈${fmt(res.noiseEstimate, 4)}`;
  const lo = document.getElementById('lambdaOut');
  if (lo && $('#lambdaMode').value === 'auto') lo.textContent = 'auto (' + fmt(res.assumption.lambda, 4) + ')';

  $('#imgObs').src = res.images.refObs;
  $('#imgEst').src = res.images.estimate;
  $('#imgPred').src = res.images.predictedRef;
  $('#imgRes').src = res.images.residualCombined;
  const sig = res.noiseEstimate;
  if (sig) $('#capRes').textContent = `|input − prediction| · mean over frames · colour 0 → ${(4 * sig).toFixed(3)} (4σ)`;
  renderSupport(res);
  renderResidualMetrics(res);
  renderKernel(res);
  renderHypotheses(res);
  loadTruthForEval(res);
  applyDisplayGain();
}

function applyDisplayGain() {
  const g = Number($('#dispGain').value);
  $('#dispGainOut').textContent = g.toFixed(2);
  // Canvas-based gain because <img> cannot filter pixel values exactly.
  for (const id of ['imgObs', 'imgEst']) {
    const img = document.getElementById(id);
    if (!img.src) continue;
    regradeImage(img, g);
  }
}

const regradeCache = new Map();
function regradeImage(imgEl, gain) {
  if (!imgEl.dataset.base) return;
  const img = new Image();
  img.onload = () => {
    const cv = document.createElement('canvas');
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, cv.width, cv.height);
    for (let i = 0; i < d.data.length; i += 4) {
      d.data[i] = clamp8(d.data[i] * gain);
      d.data[i + 1] = clamp8(d.data[i + 1] * gain);
      d.data[i + 2] = clamp8(d.data[i + 2] * gain);
    }
    ctx.putImageData(d, 0, 0);
    imgEl.src = cv.toDataURL('image/png');
  };
  img.src = imgEl.dataset.base;
}

function clamp8(v) { return Math.max(0, Math.min(255, Math.round(v))); }

function renderSupport(res) {
  const img = new Image();
  img.onload = () => {
    // overlay support as a warm tint on the estimate
    const est = new Image();
    est.onload = () => {
      const cv = document.createElement('canvas');
      cv.width = est.naturalWidth; cv.height = est.naturalHeight;
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(est, 0, 0);
      const sc = document.createElement('canvas');
      sc.width = img.naturalWidth; sc.height = img.naturalHeight;
      const sctx = sc.getContext('2d', { willReadFrequently: true });
      sctx.drawImage(img, 0, 0);
      const sd = sctx.getImageData(0, 0, sc.width, sc.height);
      const ed = ctx.getImageData(0, 0, cv.width, cv.height);
      for (let i = 0; i < ed.data.length; i += 4) {
        const support = sd.data[i] / 255;
        // low support -> tint toward amber
        const t = 1 - support;
        ed.data[i] = clamp8(ed.data[i] * (1 - 0.28 * t) + 210 * 0.28 * t);
        ed.data[i + 1] = clamp8(ed.data[i + 1] * (1 - 0.34 * t) + 150 * 0.34 * t);
        ed.data[i + 2] = clamp8(ed.data[i + 2] * (1 - 0.40 * t) + 60 * 0.40 * t);
      }
      ctx.putImageData(ed, 0, 0);
      $('#imgSupportOverlay').src = cv.toDataURL('image/png');
    };
    est.src = res.images.estimate;
  };
  img.onerror = () => { $('#imgSupportOverlay').src = res.images.estimate; };
  img.src = res.images.supportMap;
  $('#imgSupport').src = res.images.supportMap;
}

function renderResidualMetrics(res) {
  const st = res.residualStats;
  const ev = res.evidence;
  const fwdToNoise = ev.forwardToNoise;
  const cls = fwdToNoise < 1.4 ? 'good' : fwdToNoise < 3 ? 'warn' : 'bad';
  const items = [
    { v: fmt(st.rmse, 4), l: 'forward RMSE', cls },
    { v: fmt(fwdToNoise, 2) + '×', l: 'fwd / noise σ', cls },
    { v: fmt(ev.noiseSigmaEstimate, 4), l: 'noise σ (estimated)', cls: '' },
    { v: fmt(st.peak, 3), l: 'peak |residual|', cls: '' },
  ];
  $('#residualMetrics').innerHTML = items.map((m) =>
    `<div class="metric ${m.cls}" role="listitem"><div class="v">${m.v}</div><div class="l">${m.l}</div></div>`).join('');
  const verdict = fwdToNoise < 1.4
    ? 'The re-blurred estimate explains the input at the noise floor: the assumed model is consistent with the data.'
    : fwdToNoise < 3
      ? 'The residual is above the noise floor. The kernel/shutter assumption is only approximately consistent; some structure is unexplained.'
      : 'The residual is far above the noise floor. This assumption does NOT explain the input — change the motion/shutter model before trusting the estimate.';
  $('#residualNote').innerHTML =
    `${verdict}<br><span class="hint">Residual is computed by applying the assumed exposure operator to the estimate and comparing with the observed frame(s). No ground truth is involved.</span>`;
}

function renderKernel(res) {
  const k = res.kernel;
  const cv = $('#kernelCanvas');
  const ctx = cv.getContext('2d');
  const { size, data, radius } = k;
  const scale = Math.floor(cv.width / size) || 1;
  let max = 0;
  for (const v of data) max = Math.max(max, v);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = max > 0 ? Math.pow(data[y * size + x] / max, 0.65) : 0;
      const r = Math.round(255 * Math.min(1, v * 1.4));
      const g = Math.round(255 * Math.min(1, v * 1.05));
      const b = Math.round(255 * Math.min(1, 0.3 + v * 0.7));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }
  // centre cross
  ctx.strokeStyle = '#0b111b'; ctx.lineWidth = 1;
  const c = Math.floor(size / 2) * scale;
  ctx.beginPath(); ctx.moveTo(c + scale / 2, 0); ctx.lineTo(c + scale / 2, cv.height); ctx.stroke();
  const a = res.assumption;
  $('#kernelMeta').innerHTML = `
    <div>assumed path <b>${a.kind}</b></div>
    <div>velocity <b>(${fmt(a.vx, 1)}, ${fmt(a.vy, 1)})</b> px/frame</div>
    <div>shutter ρ <b>${fmt(a.rho, 2)}</b> · κ(t) <b>${a.exposureKind}</b></div>
    <div>kernel length ≈ <b>${fmt(Math.hypot(a.vx, a.vy) * a.rho, 1)}</b> px</div>
    <div>grid <b>${size}×${size}</b> (r=${radius})</div>
    <div class="hint" style="margin-top:4px">This kernel is derived from assumptions, not from truth.</div>`;
}

function renderHypotheses(res) {
  const body = $('#hypBody');
  const cmp = res.comparison;
  if (!cmp) {
    body.innerHTML = `<p class="note">Hypothesis comparison was not requested. Tick “compare hypotheses” and invert again.</p>`;
    return;
  }
  const max = Math.max(...cmp.scored.map((s) => s.forwardRmse), 1e-9);
  const rows = cmp.scored.map((s, i) => `
    <div class="hyp-row ${i === 0 ? 'best' : ''}">
      <div class="nm"><b>${s.label}</b><span class="hint">${s.id}</span></div>
      <div class="barwrap"><span style="width:${(s.forwardRmse / max) * 100}%"></span></div>
      <div class="num">${fmt(s.forwardRmse, 4)}</div>
    </div>`).join('');
  const m = cmp.margin;
  const ambiguous = cmp.ambiguous;
  const verdict = ambiguous
    ? `<div class="hyp-verdict ambiguous"><b>Ambiguous — both kept.</b> The top two explanations differ by ${fmt(m * 100, 1)}% in forward residual (threshold 5%). The data cannot separate them; both estimates are valid explanations of the same blurred input. Use inter-frame motion (registration) to resolve direction.</div>`
    : `<div class="hyp-verdict resolved"><b>Resolved.</b> “${cmp.scored[0].label}” explains the input ${fmt(m * 100, 0)}% better than the runner-up by forward residual.</div>`;
  const thumbs = cmp.results.map((r) => `
    <figure>
      <img src="${r.estimateImage}" alt="${escapeHTML(r.label)}" />
      <figcaption style="${cmp.scored[0].id === r.id ? 'color:var(--good)' : ''}">${escapeHTML(r.label)} · ${fmt(r.forwardRmse, 4)}</figcaption>
    </figure>`).join('');
  body.innerHTML = `<div>${rows}</div>${verdict}
    <p class="hyp-note">Bars show forward RMSE (shorter is better). Registration from the player: vx ${fmt(state.clip.registration.candidate.vx, 2)}, vy ${fmt(state.clip.registration.candidate.vy, 2)}.</p>
    <div class="thumb-row">${thumbs}</div>`;
}

async function loadTruthForEval(res) {
  // Ground truth for scoring only, clearly labelled.
  try {
    if (!state.truth) state.truth = await getJSON(`/api/truth?id=${encodeURIComponent(state.clipId)}`);
  } catch (e) {
    $('#evalBody').innerHTML = `<p class="note">truth unavailable: ${e.message}</p>`;
    return;
  }
  const ev = res.evidence;
  const rows = [
    ['PSNR (estimate vs truth)', fmt(ev.psnr, 2) + ' dB', ''],
    ['PSNR (blurred input vs truth)', fmt(ev.blurPsnr, 2) + ' dB', ''],
    ['gain over blurred input', '+' + fmt(ev.gainDb, 2) + ' dB', ev.gainDb > 3 ? 'good' : 'warn'],
    ['SSIM', fmt(ev.ssim, 3), ev.ssim > 0.8 ? 'good' : 'warn'],
    ['forward RMSE', fmt(ev.forwardRmse, 4), ''],
    ['noise σ (estimated, no truth)', fmt(ev.noiseSigmaEstimate, 4), ''],
    ['ringing index', fmt(ev.ringing.index, 1), ev.ringing.index < 8 ? 'good' : 'warn'],
    ['overshoot fraction', fmt(ev.ringing.overshootFraction * 100, 1) + '%', ev.ringing.overshootFraction < 0.01 ? 'good' : 'warn'],
  ];
  const regionRows = ev.regions.map((r) =>
    `<tr><td>${r.label}</td><td class="num">${fmt(r.rmse, 4)}</td><td class="num">${fmt(r.psnr, 1)}</td><td class="num">${fmt(r.estStd, 3)} / ${fmt(r.truthStd, 3)}</td><td class="num" style="color:${r.detailCorr > 0.6 ? 'var(--good)' : r.detailCorr > 0.3 ? 'var(--warn)' : 'var(--bad)'}">${fmt(r.detailCorr, 2)}</td></tr>`).join('');
  const tex = ev.regions.find((r) => r.label.includes('texture'));
  const flat = ev.regions.find((r) => r.label.includes('flat'));
  const texVerdict = !tex ? '' : tex.detailCorr > 0.6
    ? `Fine-texture detail here correlates with the truth (r=${fmt(tex.detailCorr, 2)}) — the multi-frame sampling really did recover this band.`
    : `Fine-texture detail here does <b>not</b> correlate with the truth (r=${fmt(tex.detailCorr, 2)}); whatever sharpness is visible there is prior fill, not measured detail.`;
  const flatVerdict = flat && flat.estStd > 2.5 * flat.truthStd
    ? ` The flat area shows ${fmt(flat.estStd / Math.max(flat.truthStd, 1e-3), 1)}× the true variation, so the extra texture is solver noise.`
    : '';
  $('#evalBody').innerHTML = `
    <table class="metrics"><tbody>
      ${rows.map(([k, v, c]) => `<tr><td>${k}</td><td class="num" style="color:${c === 'good' ? 'var(--good)' : c === 'warn' ? 'var(--warn)' : 'inherit'}">${v}</td></tr>`).join('')}
    </tbody></table>
    <h3 style="margin:12px 0 6px">Per-region error and detail correlation</h3>
    <table class="metrics"><thead><tr><th>region</th><th>RMSE</th><th>PSNR</th><th>std est/truth</th><th>detail r</th></tr></thead><tbody>${regionRows}</tbody></table>
    <div class="compare-grid two" style="margin-top:12px">
      <figure><img src="${state.truth.latent}" alt="truth latent" /><figcaption>ground truth latent</figcaption></figure>
      <figure><img src="${res.images.supportMap}" alt="support" /><figcaption>data support (for reference)</figcaption></figure>
    </div>
    <p class="note">${escapeHTML(ev.note || '')} ${texVerdict}${flatVerdict} <b>detail r</b> is the correlation of high-pass content with the truth: matching variance alone can be faked, matching detail cannot.</p>`;
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------------------
// Tabs + blink
// ---------------------------------------------------------------------------

function wireTabs() {
  $$('.tab').forEach((t) => t.addEventListener('click', () => {
    $$('.tab').forEach((x) => x.classList.toggle('active', x === t));
    $$('.tab-pane').forEach((p) => p.classList.toggle('active', p.dataset.pane === t.dataset.tab));
  }));
}

let blinkTimer = null;
function wireResultControls() {
  $('#blink').addEventListener('change', (e) => {
    if (blinkTimer) clearInterval(blinkTimer);
    blinkTimer = null;
    if (e.target.checked) {
      let flip = false;
      blinkTimer = setInterval(() => {
        flip = !flip;
        const img = $('#imgEst');
        if (!img) return;
        img.style.opacity = flip ? '0.12' : '1';
      }, 420);
    } else {
      $('#imgEst').style.opacity = '1';
    }
  });
}

// ---------------------------------------------------------------------------
// Build deliverable videos
// ---------------------------------------------------------------------------

async function runBuild() {
  if (state.build.running) return;
  state.build.running = true;
  const btn = $('#btnBuild');
  btn.disabled = true;
  btn.textContent = 'building…';
  setJobStatus('running', 'building deliverable videos', 'this may take a few minutes');
  try {
    const r = await fetch('/api/build', { method: 'POST' });
    if (!r.ok) throw new Error(`build → ${r.status}`);
    const { jobId } = await r.json();
    for (;;) {
      await sleep(800);
      const j = await getJSON(`/api/build/status?id=${jobId}`);
      setJobStatus(j.state === 'failed' ? 'failed' : j.state === 'done' ? 'done' : 'running', j.stage || j.state, j.detail || '');
      if (j.state === 'done') {
        $('#sbJob').textContent = `build: done → /out`;
        setJobStatus('done', 'videos ready', 'open /out/report.md');
        break;
      }
      if (j.state === 'failed') {
        setJobStatus('failed', 'build failed', j.error || '');
        break;
      }
    }
  } catch (e) {
    setJobStatus('failed', 'build error', String(e.message || e));
  } finally {
    state.build.running = false;
    btn.disabled = false;
    btn.textContent = 'Build deliverable videos';
  }
}
