# Zhou 2020 Validation Figure Contract

Core conclusion:
The signed FEniCSx responses must agree with the independent empymod layered-earth reference over the complete prescribed time window before the polarization implementation can be considered validated.

Figure archetype:
Quantitative grid with one dominant signed-response comparison and subordinate error, IP-increment, and convergence evidence.

Target journal/output:
Technical validation report and manuscript-ready supporting figure.

Backend:
Python (matplotlib) exclusively.

Final size:
183 mm double-column width; SVG and PDF with editable text, 600 dpi TIFF, and 300 dpi PNG for Word.

Panel map:
- a: Zhou grounded-wire layered model, source, receiver, and coordinate convention.
- b: signed no-IP Ex comparison over the complete time window.
- c: signed no-IP Hz and dBz/dt comparisons over the complete time window.
- d: component-wise full-window relative L2 errors and pass/fail gates.
- e: signed IP and no-IP responses plus IP increments.
- f: first zero-crossing and adjacent spatial, time-step, and boundary convergence.

Evidence hierarchy:
- Hero evidence: complete signed FEniCSx-versus-empymod response curves.
- Validation evidence: full-window relative L2 metrics and zero-crossing timing.
- Controls/robustness: no-IP gate, independent source-quadrature convergence, spatial/time/boundary convergence, full sample-count checks, and hash-backed manifests.

Statistics needed:
No inferential statistics. Report deterministic relative L2 norms, robust relative changes, zero-crossing timing, sample counts, and numerical thresholds.

Source data needed:
The immutable empirical reference CSVs, immutable FEniCSx predictions, comparison JSON, convergence JSON, case YAML, provenance JSON, and SHA-256 manifests.

Image-integrity notes:
All 101 prescribed samples must be plotted. No smoothing, clipping, selective omission, manual sign correction, or post-hoc baseline adjustment is permitted. Signed data use symlog axes; absolute values may appear only in explicitly labeled error panels.

Reviewer risk:
- The journal article is the authoritative methodological citation, while numerical model values not exposed in the article metadata are traced to the same inventors' companion patent and must be labeled as such.
- empymod uses a 1D layered-earth solution and a 0.1 m numerical surface offset; the offset sensitivity must remain documented.
- dB/dt for a step-off source must be computed as minus permeability times the impulse H response, not as a step-off B response.
- A matching Hz curve alone is insufficient because the selected benchmark has weak magnetic IP sensitivity; Ex and dBz/dt must also pass.
- Any failed gate remains visible and prevents progression to the IP acceptance claim.
