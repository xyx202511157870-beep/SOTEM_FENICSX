# Channel anomaly diagnostic figures design

## Goal

Make the finite seepage channel visible without removing the conventional total-field and absolute-amplitude decay plots already approved for the Word report.

## Considered approaches

1. Add only a relative-anomaly decay plot. This is compact, but it does not preserve polarity or show the spatial receiver pattern.
2. Add relative-anomaly decay, signed anomaly, and representative-time spatial profiles. This is the recommended and approved approach because each figure answers a different diagnostic question.
3. Replace the existing absolute-amplitude anomaly plot with a normalized plot. This shortens the report, but it loses continuity with the previously approved conventional-decay presentation.

The report will use approach 2 and retain the existing figures.

## Data and formulas

- Read the existing `benchmark_results.npz`; do not rerun or alter SimPEG, FEniCSx, or empymod solver outputs.
- Use the stored signed differences `simpeg_delta` and `fenicsx_delta`, where
  `delta = channel total field - background total field`.
- Compute the pointwise relative anomaly as
  `100 * abs(delta) / max(abs(background), floor)`.
- Define `floor` separately for every receiver and component as
  `1e-12 * max_t(abs(background))`, with the smallest positive floating-point value as an absolute lower bound. This prevents division by zero without changing normal strong-signal ratios.
- Continue to use only reporting receivers Rx1, Rx2, Rx4, and Rx5. Rx3 remains in raw NPZ/CSV artifacts but is excluded from every new formal figure.

## Figures

### Relative-anomaly decay

- Add `channel_relative_anomaly.png` and `.pdf`.
- Use three panels for Ex, dBz/dt, and Hz.
- Use logarithmic time and logarithmic percentage axes.
- Plot SimPEG and FEniCSx for the four reporting receivers.
- Label the vertical axis `relative anomaly (%)` and state the formula in the Word text.

### Signed anomaly

- Add `channel_delta_signed.png` and `.pdf`.
- Use three panels for Ex, dBz/dt, and Hz.
- Use logarithmic time and symmetric-log response axes.
- Preserve the original sign and zero crossings from the stored delta arrays.
- This figure supplements, rather than replaces, the existing absolute-amplitude `channel_delta` decay figure.

### Representative-time spatial profiles

- Add `channel_relative_anomaly_profiles.png` and `.pdf`.
- Use the nearest available output gates to `1e-5 s`, `3.162e-4 s`, and `1e-2 s`.
- Use receiver physical `y` position on the horizontal axis and relative anomaly percent on the vertical axis.
- Use three component panels, gate colors, method line styles, and only the four reporting receivers.
- This figure must make the stronger response at the inner receiver pair distinguishable from the outer pair without claiming a continuous receiver survey between the four samples.

## Word report integration

- Preserve the current background, total-channel, and absolute channel-anomaly decay figures.
- Insert the relative-anomaly, signed-anomaly, and spatial-profile figures after the existing absolute channel-anomaly figure.
- Renumber the current channel-delta algorithm-error figure after the three new figures.
- Add concise text explaining that the total field is background dominated, the relative plot reveals anomaly strength, the signed plot preserves polarity and zero crossings, and the spatial profile shows receiver-position dependence.
- Keep all solver parameters, source geometry, z-down convention, full-domain FEniCSx statement, and raw-data inventory unchanged.

## Testing and verification

- Test the relative-anomaly calculation, including an exact expected ratio and finite output for a zero background sample.
- Test that the new figure stems are part of the formal artifact set.
- Test that the signed figure retains negative values on symmetric-log axes.
- Test that the spatial-profile gate selection and receiver set exclude Rx3.
- Test the Word source for the new headings, captions, formula explanation, and updated figure numbering.
- Regenerate all plots and the DOCX, render the DOCX to PDF/pages, and visually inspect every page for clipping, overlap, legibility, and caption placement.

## Scope boundaries

- No solver rerun, mesh change, conductivity change, receiver relocation, or time-grid change.
- No use of one-sided FEniCSx values or mirrored field values.
- No modification of signed CSV/NPZ data.
- No claim that empymod solves the finite 3D channel; it remains the background-only 1D reference.
