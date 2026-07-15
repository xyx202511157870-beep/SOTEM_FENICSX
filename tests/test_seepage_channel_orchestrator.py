from tools.run_seepage_channel_benchmark import build_run_plan


def test_run_plan_contains_no_mirrored_fenicsx_jobs() -> None:
    plan = build_run_plan(output_root="output/seepage_channel_100m_5rx")
    assert [job.name for job in plan] == [
        "empymod_background",
        "simpeg_background",
        "simpeg_channel",
        "fenicsx_background",
        "fenicsx_channel",
        "aggregate",
        "plot",
        "manifest",
    ]
    assert all("mirror" not in " ".join(job.command).lower() for job in plan)
