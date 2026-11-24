# Blur Inversion Studio

Multi-frame exposure-blur inversion prototype. Given a set of *blurred* frames and
an explicit, user-stated motion + shutter model, it estimates a sharper latent image
by running a real regularised least-squares solve (preconditioned conjugate
gradients), then re-blurs that estimate through the *same* forward model and shows
the residual as evidence of whether the assumption actually explains the input.

Ground truth is synthetic. It never reaches the solver; it is only used, separately,
to score the result (PSNR / SSIM / per-region error) and that scoring is labelled as
such in the UI.

## Launch the UI

```bash
npm install          # installs ffmpeg-static (used only for MP4 export)
npm start            # serves the studio on http://localhost:5173
```

Then open **http://localhost:5173**. No build step is required (the front end is
plain ES modules served statically).

Other commands:

```bash
npm test             # 20 numerical kernel tests (no browser needed)
npm run build        # regenerate out/*.mp4, comparison PNGs, report.json/report.md
npm run smoke        # headless-browser UI walkthrough; writes screenshots to test-output/
                     # (needs Playwright + system Chromium libs; not required to use the UI)
```

`npm run build` writes the deliverable videos and the metrics report into `out/`
and `report.md`; the same files are downloadable from the UI's **Build deliverable
videos** button.

## How to use it

1. **1 · Blurred clips** — pick a synthetic clip. These are the observations; the
   clean sequence is *not* shown by default and is never used by the solver.
2. **2 · Motion analysis player** — scrub / play the blurred frames. *show clean
   reference* reveals the truth for comparison only, flagged as player-only. The
   blind phase-correlation plot estimates inter-frame motion from the observations;
   blur biases this estimate toward zero, so it is a starting point, not truth.
3. **3 · Motion & shutter assumption** — state the model explicitly:
   - motion kind (constant velocity / shake / arc) and vx, vy,
   - shutter fraction ρ and exposure weighting κ(t) (uniform / centred / ramp),
   - reference frame, regularisation λ (auto from the noise estimate, or manual),
     TV weight, max CG iterations,
   - *compare hypotheses* to run the forward-residual comparison as well.
4. **4 · Inversion result** — tabs:
   - **estimate** — the latent estimate (display gain is cosmetic and never feeds
     the solver or metrics),
   - **data residual** — re-blurred estimate vs input, colour-scaled to 4σ,
   - **data support** — local share of high-frequency energy the assumed kernels
     still pass; this separates measured detail from prior fill,
   - **exposure kernel** — the assumed per-frame kernels and displacements,
   - **hypotheses** — forward RMSE for each motion hypothesis; if the top two are
     within 5% the ambiguity is kept and both are reported rather than one being
     silently chosen,
   - **evaluation\*** — ground-truth scoring (PSNR/SSIM/ringing/per-region std);
     marked as evaluation-only.

When an assumption fails (forward residual far above the noise floor, or the solve
diverging) the failure is reported in the result area and fed back as guidance to
change the motion/shutter model — it is not hidden behind a "looks sharper" image.

## Model

Forward model per frame k, shared by generation and inversion:

```
g_k = crop( S_{d_k}( h_k * x ) )
```

- `x` is the latent on a padded 256×128 canvas (zero-padded, circular convolution),
- `h_k` is the exposure kernel: the path of the global motion, integrated over the
  shutter with weight κ(t), and truncated to shutter fraction ρ,
- `S_{d_k}` is an exact Fourier sub-pixel shift to frame k's position relative to
  the reference,
- `crop` extracts the observed 160×96 window,
- observations are quantised to 8-bit and optionally have Gaussian noise added.

The inversion minimises

```
Σ_k || W_k ( A_k x − g_k ) ||²  +  λ || L x ||²  +  λ_tv · TV(x)
```

where `A_k = S_{d_k} H_k`, `W_k` is an observation-support mask, and `L` is a
high-pass (Laplacian) prior. It is solved by preconditioned CG with a logged
residual history — a real iterative optimisation, not a sharpening filter.

### Scale / shift ambiguity

The exposure kernel and the latent image share a scale and a position: a wider
kernel looks like a more blurred latent, and a global shift can be absorbed into
both. The constraints used here:

- the latent lives on a fixed padded canvas with a fixed observation crop, so the
  observation coordinate frame is pinned to the reference frame,
- inter-frame displacements `d_k` are taken from the stated motion path relative to
  the reference frame, not estimated freely during the solve,
- the shutter fraction ρ and exposure weighting κ(t) are explicit inputs, so the
  kernel's width is not a free parameter of the inversion,
- λ is fixed (auto from the noise estimate or manual), so the prior strength does
  not silently trade against kernel width.

### Data-supported vs prior-filled

`ρ(f) = Σ_k |A_k(f)|² / (Σ_k |A_k(f)|² + λ|L(f)|²)` is the frequency-domain
reliability of each mode. The support map averages this over local windows of the
estimate's high-frequency energy. Values near 1 mean the observations constrain
those frequencies; low values mean the prior is choosing. It uses only the
observations, the assumed operators and λ — no truth.

Fine texture beyond the kernel cutoff is **not** recovered. The evaluation tab
reports `est std / truth std` per region: if that ratio is well below 1 in the
fine-texture patch, the sharpness you see there is prior fill, not measured detail.
The UI says this explicitly rather than treating extra local contrast as success.

## Results

`npm run build` produces `out/report.md` and `out/report.json`, plus a labelled
four-panel MP4 per case (clean sequence / blurred input / inverted estimate /
forward residual) and still comparison sheets. Example (synthetic, 7 frames):

| clip | blur PSNR | estimate PSNR | gain | SSIM | forward RMSE | fwd/σ | ringing |
|---|---|---|---|---|---|---|---|
| pan-clean | 28.6 | 40.5 | +11.9 dB | 0.965 | 0.0008 | 0.92 | 1.9 |
| diag | 27.5 | 39.4 | +11.9 dB | 0.922 | 0.0030 | 0.96 | 2.5 |
| fast | 24.6 | 32.3 | +7.8 dB | 0.806 | 0.0033 | 1.12 | 8.5 |
| noisy (σ 0.012) | 27.4 | 31.0 | +3.6 dB | 0.736 | 0.0116 | 1.06 | 11.7 |

`fwd/σ ≈ 1` means the re-blurred estimate explains the input down to the noise
floor; a reversed or stationary kernel gives 25–46× larger forward residuals and is
rejected by the same test.

## Limitations

- Restricted to global motion that can be described by the parametric paths offered
  (linear, shake, arc). Local/independent motion is out of scope.
- The motion parameters are user-stated assumptions; blind registration only
  provides a starting point and is biased toward zero blur.
- Low-texture regions and severe blur can have no unique solution; the support map
  and per-region std ratio expose this instead of inventing detail.
- The synthetic scene and forward model are discrete and shared between generation
  and inversion, so the clean-case forward residual is exactly zero. Real sensor
  data would add calibration error that this prototype does not model.

## Files

```
src/fft.js        radix-2 real 2D FFT, Fourier shift, padding helpers
src/synth.js      scene, motion presets, exposure kernels, clip synthesis
src/model.js      forward operator spectra and frame kernels
src/motion.js     phase correlation and blind registration
src/solve.js      regularised CG solve, support map, re-blur residual
src/pipeline.js   analyse → build operators → invert → compare hypotheses
src/evaluate.js   the only module that touches truth (scoring only)
server/server.js  HTTP API + static serving + job runner
server/media.js   PNG encoder and ffmpeg MP4 export
server/artifacts.js  offline video/report builder
web/              the studio UI (index.html, app.js, styles.css)
scripts/test.js   numerical kernel tests
scripts/ui-smoke.mjs  headless-browser walkthrough
```
