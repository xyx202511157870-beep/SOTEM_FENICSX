# Channel response conventional decay plots design

## Goal

Convert Word-report Figures 4 and 5 to conventional transient-EM absolute-amplitude decay plots.

## Design

- Enable the existing `magnitude_decay=True` mode for `channel_response` and `channel_delta`.
- Plot absolute Ex, dBz/dt, and Hz amplitudes on log-log axes.
- Keep the four reporting receivers Rx1, Rx2, Rx4, and Rx5.
- Preserve the original signed channel and channel-minus-background arrays in CSV/NPZ files.
- Change Figure 4 caption to "channel-model absolute-amplitude decay curves".
- Change section 4.3 and Figure 5 wording from "signed anomaly" to "absolute-amplitude anomaly decay curves".
- Do not change Figure 6 error curves or magnetic-method audit plots.

## Verification

- A call-level test must prove all three response-grid calls enable magnitude decay.
- Word tests must prove the new Figure 4 and Figure 5 wording is present and old signed-anomaly wording is absent.
- Regenerate and visually inspect the DOCX/PDF.

