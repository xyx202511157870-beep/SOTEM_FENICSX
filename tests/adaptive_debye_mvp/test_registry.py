from atem3d.adaptive_debye_mvp.candidates import generate_candidates
from atem3d.adaptive_debye_mvp.protocol_constants import SPLIT_COUNTS
from atem3d.adaptive_debye_mvp.registry import (
    freeze_case_registry,
    generate_all_cases,
    instantiate_candidates,
    protocol_candidate_config,
)


def test_case_registry_is_deterministic_and_complete():
    left = generate_all_cases()
    right = generate_all_cases()
    assert [case.case_id for case in left] == [case.case_id for case in right]
    assert [case.case_hash() for case in left] == [case.case_hash() for case in right]
    assert len(left) == 40
    counts = {split: sum(case.split == split for case in left) for split in SPLIT_COUNTS}
    assert counts == SPLIT_COUNTS


def test_independent_test_is_not_a_train_repeat():
    cases = generate_all_cases()
    train = [case for case in cases if case.split == "train"]
    test = [case for case in cases if case.split == "independent_test"]
    train_keys = {(round(case.m, 8), round(case.tau, 12), round(case.c, 8)) for case in train}
    for case in test:
        assert (round(case.m, 8), round(case.tau, 12), round(case.c, 8)) not in train_keys
        assert case.sensor_frame == "tilted"
        assert "W3" in case.waveform_ids
        assert case.source_azimuth_deg >= 90.0


def test_candidate_instantiation_does_not_invent_new_ids():
    cases = generate_all_cases()
    catalog = {spec.candidate_id for spec in generate_candidates(protocol_candidate_config())}
    for case in cases[:3]:
        instantiated = instantiate_candidates(case)
        assert {spec.candidate_id for spec in instantiated} == catalog
        assert all(spec.K in {4, 6, 8, 10, 12} for spec in instantiated)


def test_freeze_case_registry_roundtrip(tmp_path):
    path = tmp_path / "case_registry.csv"
    written = freeze_case_registry(path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("case_id,split,split_index")
    assert len(written) == 40
