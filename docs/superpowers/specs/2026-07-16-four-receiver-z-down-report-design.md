# Four-receiver z-down report design

## Goal

Revise the formal Word report so that physical depth increases downward in every geometry panel and the center receiver directly beneath the source wire is excluded from the reporting view.

## Scope and data policy

- Keep the completed five-receiver solver outputs, checkpoint data, CSV/NPZ artifacts, and magnetic stability audit unchanged.
- Define the formal reporting receiver indices as `(0, 1, 3, 4)`, corresponding to the existing labels `Rx1`, `Rx2`, `Rx4`, and `Rx5`.
- Do not renumber the retained receivers because their labels must continue to map directly to the raw arrays.
- Exclude the center receiver from the model geometry, response curves, error curves, receiver table, and magnetic-report figures.
- Retain a short provenance statement that the raw solve evaluated five points but the formal report presents four usable non-center receivers.

## Geometry presentation

- The coordinate convention remains `z=0` at the surface and positive `z` underground.
- Invert the displayed z direction in the 3D panel as well as both 2D sections so that greater physical depth appears lower on the page.
- Show only the four reporting receivers in the 3D and y-z panels.

## Word report presentation

- Change all formal observation-system wording and captions from five displayed receivers to four reporting receivers.
- Remove the center-point residual plot and center-point acceptance row from the magnetic stability section.
- Keep the selected magnetic method and the two retained odd-pair residuals so the method decision remains documented.
- Keep the full-domain and no-mirroring statements, explicitly distinguishing raw solver provenance from the four-point report view.

## Verification

- Automated tests must prove that the geometry uses a downward display axis and contains four receiver markers.
- Automated tests must prove that response and error plots contain no center-receiver curves.
- The Word test must prove that the report names the four retained receivers, omits the center-residual figure, and retains full-domain provenance.
- Regenerate the DOCX, export it to PDF, render every page, and visually inspect the layout.

