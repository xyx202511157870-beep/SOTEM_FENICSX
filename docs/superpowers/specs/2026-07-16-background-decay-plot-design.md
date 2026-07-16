# Background response conventional decay plot design

## Goal

Render Word-report Figure 2 as a conventional transient-EM decay plot instead of a signed symmetric-log plot.

## Design

- Apply the change only to `background_response`.
- Plot `abs(response)` for Ex, dBz/dt, and Hz.
- Use logarithmic time and logarithmic amplitude axes.
- Label each vertical axis as an absolute magnitude so the removed display sign is explicit.
- Retain the existing four reporting receivers Rx1, Rx2, Rx4, and Rx5.
- Preserve all signed arrays in CSV/NPZ artifacts.
- Keep `channel_response` and `channel_delta` on their current signed presentation because anomaly polarity remains meaningful there.
- Update the Word caption to say "absolute-amplitude decay curves".

## Verification

- A plotting test must fail before implementation when the background figure still uses `symlog` and signed samples.
- The test must pass after the background figure uses positive absolute values and log y axes.
- Regenerate the Word report, export to PDF, and visually inspect all pages.

