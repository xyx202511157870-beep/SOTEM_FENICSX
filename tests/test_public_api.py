from atem3d import (
    AverageReceiver,
    CorrectedModelValidationConfig,
    PointReceiver,
    build_corrected_model_case_specs,
    build_receiver,
)


def test_receiver_builders_are_available_from_public_api():
    point = build_receiver(
        location=(0.0, 0.0, 0.0),
        component="Ex",
        receiver_type="point",
    )
    average = build_receiver(
        location=(0.0, 0.0, 0.0),
        component="Ex",
        receiver_type="disk_average",
        radius=1.0,
    )

    assert isinstance(point, PointReceiver)
    assert isinstance(average, AverageReceiver)


def test_corrected_model_helpers_are_available_from_public_api(tmp_path):
    config = CorrectedModelValidationConfig()
    specs = build_corrected_model_case_specs(tmp_path, config)

    assert specs["noip"]["source_start"] == [-500.0, 200.0, -0.1]
