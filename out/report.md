# Blur inversion — synthetic results

Ground truth is synthetic and used for scoring only. The solver receives the blurred observations and the stated motion/shutter assumptions.

| clip | blur PSNR | estimate PSNR | gain | SSIM | forward RMSE | noise σ | fwd/σ | ringing | reversed fwd RMSE |
|---|---|---|---|---|---|---|---|---|---|
| pan-clean | 28.62 | 40.52 | +11.89 dB | 0.965 | 0.0008 | 0.0008 | 0.92 | 1.9 | 0.1226 |
| pan-long | 28.07 | 33.00 | +4.93 dB | 0.842 | 0.0035 | 0.0033 | 1.08 | 9.2 | 0.1209 |
| diag | 27.54 | 39.40 | +11.86 dB | 0.922 | 0.0030 | 0.0031 | 0.96 | 2.5 | 0.1350 |
| fast | 24.58 | 32.33 | +7.75 dB | 0.806 | 0.0033 | 0.0030 | 1.12 | 8.5 | 0.1557 |
| shake | 26.72 | 39.16 | +12.44 dB | 0.927 | 0.0032 | 0.0030 | 1.08 | 3.2 | 0.1081 |
| noisy | 27.39 | 30.95 | +3.56 dB | 0.736 | 0.0116 | 0.0110 | 1.06 | 11.7 | 0.1214 |

`fwd/σ` near 1 means the re-blurred estimate explains the input down to the noise floor; well above 1 means the assumed model does not explain the data.

Per-region detail: `std` is the standard deviation of the estimate vs the truth in that region; `detail r` is the correlation of high-pass content. Matching variance alone can be faked by noise, so the correlation is the honest test of whether detail was recovered.

| clip | region | RMSE | PSNR | std est/truth | detail r |
|---|---|---|---|---|---|
| pan-clean | bar chart (recoverable detail) | 0.0109 | 39.2 | 0.240 / 0.242 | 0.99 |
| pan-clean | fine texture patch (near Nyquist) | 0.0271 | 31.3 | 0.039 / 0.059 | 0.95 |
| pan-clean | flat area (low texture) | 0.0100 | 40.0 | 0.012 / 0.008 | 0.05 |
| pan-long | bar chart (recoverable detail) | 0.0492 | 26.2 | 0.233 / 0.242 | 0.77 |
| pan-long | fine texture patch (near Nyquist) | 0.0508 | 25.9 | 0.028 / 0.059 | 0.38 |
| pan-long | flat area (low texture) | 0.0220 | 33.1 | 0.023 / 0.008 | 0.01 |
| diag | bar chart (recoverable detail) | 0.0132 | 37.6 | 0.228 / 0.230 | 0.97 |
| diag | fine texture patch (near Nyquist) | 0.0089 | 41.0 | 0.062 / 0.063 | 0.99 |
| diag | flat area (low texture) | 0.0093 | 40.6 | 0.010 / 0.004 | 0.05 |
| fast | bar chart (recoverable detail) | 0.0468 | 26.6 | 0.216 / 0.243 | 0.86 |
| fast | fine texture patch (near Nyquist) | 0.0484 | 26.3 | 0.030 / 0.060 | 0.57 |
| fast | flat area (low texture) | 0.0166 | 35.6 | 0.017 / 0.003 | -0.00 |
| shake | bar chart (recoverable detail) | 0.0179 | 34.9 | 0.242 / 0.244 | 0.96 |
| shake | fine texture patch (near Nyquist) | 0.0090 | 40.9 | 0.054 / 0.059 | 0.99 |
| shake | flat area (low texture) | 0.0164 | 35.7 | 0.017 / 0.004 | -0.00 |
| noisy | bar chart (recoverable detail) | 0.0660 | 23.6 | 0.240 / 0.251 | 0.55 |
| noisy | fine texture patch (near Nyquist) | 0.0492 | 26.2 | 0.033 / 0.060 | 0.55 |
| noisy | flat area (low texture) | 0.0183 | 34.8 | 0.019 / 0.005 | 0.04 |