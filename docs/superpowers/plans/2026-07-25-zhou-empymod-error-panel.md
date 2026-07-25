# Zhou 2020 empymod error-panel implementation plan

1. Extend `tests/test_plot_zhou_strong_polarization_extension.py` and add
   `tests/test_plot_zhou_formal_comparison.py` so they first fail unless both
   log-response figures record the pointwise-error contract, per-condition
   denominator floors, and summary values.
2. Refactor `scripts/plot_zhou_formal_comparison.py` with a shared pointwise
   absolute-relative-error helper and replace its 1-by-3 absolute-response
   layout with the approved 2-by-3 layout.
3. Apply the same helper, four-curve response encoding, error panels, and
   multi-format export to
   `scripts/plot_zhou_strong_polarization_extension.py`.
4. Run the focused plotting tests, then regenerate both comparison directories
   directly from the existing completed NPZ files.
5. Run the nature-figure source preflight and visually inspect the regenerated
   PNG files at their final report size.
6. Update `scripts/build_zhou2020_final_validation_docx.py` captions and prose
   so that every program-generated logarithmic response figure explains its
   empymod error panels and denominator floor.
7. Rebuild the final DOCX, refresh the Word table of contents, render every page
   through Microsoft Word and Poppler, and inspect the page montage.
8. Verify the final DOCX hash, image/relationship integrity, zero user-facing
   PDFs, retained failure evidence, and unchanged completed-solver NPZ hashes.

