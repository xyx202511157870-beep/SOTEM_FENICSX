"""Dependency-free artifact-name contract shared by producers and consumers."""

REQUIRED_CASE_ARTIFACTS = (
    "predictions.csv",
    "reference_empymod_or_1d.csv",
    "errors.csv",
    "error_summary.json",
    "comparison_3comp.png",
    "error_curves_3comp.png",
    "diagnostics.json",
    "model_schematic.png",
    "run_config_resolved.yaml",
)
REQUIRED_POLARIZATION_EFFECT_ARTIFACTS = (
    "polarization_effect_predictions.csv",
    "polarization_effect_reference.csv",
    "polarization_effect_errors.csv",
    "polarization_effect_summary.json",
    "polarization_effect_comparison.png",
    "polarization_effect_error_curves.png",
)
