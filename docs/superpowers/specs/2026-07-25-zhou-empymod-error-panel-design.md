# Zhou 2020 response and empymod error-panel design

## Scope

Update the two program-generated logarithmic response figures in the final
Zhou 2020 validation report:

1. the formal weak-IP literature-benchmark comparison;
2. the derived strong-IP sensitivity comparison.

The published literature screenshot remains unchanged because no digitized
pointwise source data are available. Existing completed NPZ files are the only
numerical inputs; no FEniCSx or empymod forward solve is repeated.

## Figure contract

Each updated figure uses a 2-by-3 quantitative grid:

- columns: `Ex`, `Hz`, and `dBz/dt`;
- top row: log-log absolute response magnitudes;
- bottom row: pointwise absolute relative errors against empymod.

The top-row curve encoding is fixed:

- no-IP FEniCSx: solid blue;
- no-IP empymod: dashed blue;
- IP FEniCSx: solid orange;
- IP empymod: dashed orange.

The bottom row contains two positive error curves:

- no-IP FEniCSx versus no-IP empymod: blue;
- IP FEniCSx versus IP empymod: orange.

Every one of the 101 time samples from `1e-4` to `3 s` is retained. There is
no smoothing, cropping, or sign-dependent filtering. Existing signed-symlog
diagnostic figures remain separate so that zero crossings and signs are not
hidden by the absolute-value presentation.

## Error definition

For each component and condition:

```text
error_percent =
    100 * abs(FEniCSx - empymod)
    / max(abs(empymod), 1e-6 * peak(abs(empymod)))
```

The denominator floor prevents a reference zero crossing from producing a
meaningless numerical divergence. The floor value is computed separately for
each component and condition, recorded in the JSON metrics, and stated in the
figure caption/report text. The error axis is logarithmic because the error is
non-negative.

## Data and provenance

The formal comparison reads:

- `output/zhou_formal_s1t1b1_noip_full/verification_data.npz`
- `output/zhou_formal_s1t1b1_ip_full/verification_data.npz`

The strong-IP comparison reads:

- `output/zhou_strong_m050_h100_noip_full/verification_data.npz`
- `output/zhou_strong_m050_h100_ip8_full/verification_data.npz`

Both plotting scripts continue to write input SHA-256 hashes, sample counts,
time bounds, smoothing/cropping policy, and validation status. The report must
not change the existing fail-closed conclusion for the formal benchmark.

## Deliverables

- Updated PNG, SVG, and TIFF for the formal logarithmic response figure.
- Updated PNG, SVG, and TIFF for the strong-IP logarithmic response figure.
- Updated JSON metrics containing pointwise-error policy, floors, and summary
  statistics.
- Regenerated final Word report with revised captions and explanations.
- No PDF in the user-facing output directory.

