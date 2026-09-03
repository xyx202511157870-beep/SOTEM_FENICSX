import inspect

import numpy as np
import pytest

from atem3d.adaptive_debye_mvp.candidates import (
    FAMILY_COLE_COLE,
    FAMILY_TIME_WINDOW,
    CandidateConfig,
    freeze_candidate_registry,
    generate_candidates,
    load_candidate_registry,
    registry_hash,
    template_offsets,
    verify_candidate_registry,
)


def _default_config() -> CandidateConfig:
    return CandidateConfig(cole_cole_tau=0.1)


def test_default_config_yields_36_per_k_within_bounds():
    config = _default_config()
    candidates = generate_candidates(config)
    assert len(candidates) == 36 * 5
    for poles in config.pole_counts:
        assert sum(spec.K == poles for spec in candidates) == 36
    assert 20 <= config.candidates_per_pole_count <= 40


def test_out_of_bounds_count_rejected():
    with pytest.raises(ValueError):
        CandidateConfig(cole_cole_tau=0.1, spans=(6.0,), shifts=(0.0, 0.5))
    with pytest.raises(ValueError):
        CandidateConfig(
            cole_cole_tau=0.1,
            spans=(2.0, 4.0, 6.0, 8.0, 10.0),
            shifts=(-1.0, -0.5, 0.0, 0.5, 1.0),
        )


def test_poles_strictly_increasing_no_duplicates():
    for spec in generate_candidates(_default_config()):
        assert np.all(np.diff(spec.tau_grid) > 0.0)
        assert len(set(spec.log10_tau)) == spec.K
        assert np.all(spec.tau_grid > 0.0)


def test_same_config_hashes_identically():
    left = generate_candidates(CandidateConfig(cole_cole_tau=0.1))
    right = generate_candidates(CandidateConfig(cole_cole_tau=0.1))
    assert [spec.candidate_hash for spec in left] == [spec.candidate_hash for spec in right]
    assert registry_hash(left) == registry_hash(right)
    assert CandidateConfig(cole_cole_tau=0.1).config_hash() == CandidateConfig(cole_cole_tau=0.1).config_hash()
    shifted = generate_candidates(CandidateConfig(cole_cole_tau=0.1, shifts=(-0.25, 0.0, 0.25)))
    assert [spec.candidate_hash for spec in shifted] != [spec.candidate_hash for spec in left]


def test_offsets_symmetric_and_density():
    np.testing.assert_allclose(template_offsets(5, 4.0, 1.0), [-2.0, -1.0, 0.0, 1.0, 2.0])
    dense = template_offsets(5, 4.0, 1.25)
    np.testing.assert_allclose(dense[[0, -1]], [-2.0, 2.0])
    assert abs(dense[1]) < 1.0
    np.testing.assert_allclose(template_offsets(1, 6.0, 1.0), [0.0])


def test_families_use_expected_anchors():
    config = CandidateConfig(cole_cole_tau=0.1, pole_counts=(4, 5, 6, 8, 10))
    candidates = generate_candidates(config)
    cole = next(
        spec
        for spec in candidates
        if spec.K == 5 and spec.template_family == FAMILY_COLE_COLE and spec.shift == 0.0 and spec.span == 4.0 and spec.density_exponent == 1.0
    )
    window = next(
        spec
        for spec in candidates
        if spec.K == 5 and spec.template_family == FAMILY_TIME_WINDOW and spec.shift == 0.0 and spec.span == 4.0 and spec.density_exponent == 1.0
    )
    np.testing.assert_allclose(cole.tau_grid[2], 0.1, rtol=1.0e-9)
    np.testing.assert_allclose(window.tau_grid[2], np.sqrt(1.0e-5 * 1.0e-2), rtol=1.0e-9)


def test_candidate_ids_unique_and_deterministic():
    config = _default_config()
    first = generate_candidates(config)
    second = generate_candidates(config)
    assert len({spec.candidate_id for spec in first}) == len(first)
    assert first == second


def test_candidate_list_independent_of_data():
    parameters = inspect.signature(generate_candidates).parameters
    assert tuple(parameters) == ("config",)
    config = _default_config()
    assert generate_candidates(config) == generate_candidates(config)


def test_freeze_and_reload_registry(tmp_path):
    config = _default_config()
    path = tmp_path / "candidate_registry.csv"
    written = freeze_candidate_registry(config, path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "candidate_id,K,template_family,offsets_log10_tau,span,shift,time_window_anchor,candidate_hash"
    loaded = load_candidate_registry(path, config=config)
    assert len(loaded) == len(written)
    for left, right in zip(written, loaded):
        np.testing.assert_allclose(left.tau_grid, right.tau_grid, rtol=1.0e-9)
        assert left.candidate_hash == right.candidate_hash
    assert verify_candidate_registry(config, path)
    text = path.read_text(encoding="utf-8")
    tampered = text.replace(written[0].candidate_hash, "0" * 64, 1)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError):
        load_candidate_registry(path)
    assert verify_candidate_registry(config, path) is False
