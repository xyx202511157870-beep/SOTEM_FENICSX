import numpy as np

from atem3d.source_diagnostics import diagnose_source_consistency


def test_source_consistency_diagnostics_report_zero_residuals_for_balanced_inputs():
    gradient_transpose = np.array(
        [
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
        ]
    )
    source_vector = np.array([1.0, 1.0])
    endpoint_source = np.array([1.0, 0.0, -1.0])
    divergence_operator = np.array([[1.0, -1.0], [0.0, 1.0]])
    conductive_current = np.array([0.0, 0.0])
    initial_electric_field = np.array([2.0, 2.0])
    curl_operator = np.array([[1.0, -1.0]])

    diagnostics = diagnose_source_consistency(
        gradient_transpose=gradient_transpose,
        source_vector=source_vector,
        endpoint_source=endpoint_source,
        divergence_operator=divergence_operator,
        conductive_current=conductive_current,
        initial_electric_field=initial_electric_field,
        curl_operator=curl_operator,
        time_intervals=np.array([2.0e-6, 8.0e-6]),
        interval_average_didt=np.array([-1.0e6, -1.0e6]),
        current_initial=10.0,
        current_final=0.0,
    )

    assert diagnostics["source_endpoint_balance_residual"] == 0.0
    assert diagnostics["dc_current_conservation_residual"] == 0.0
    assert diagnostics["initial_curl_residual"] == 0.0
    assert diagnostics["waveform_integral_residual"] == 0.0


def test_source_consistency_diagnostics_measure_nonzero_residuals():
    diagnostics = diagnose_source_consistency(
        gradient_transpose=np.eye(2),
        source_vector=np.array([1.0, 0.0]),
        endpoint_source=np.array([0.0, 0.0]),
        divergence_operator=np.eye(2),
        conductive_current=np.array([2.0, -1.0]),
        initial_electric_field=np.array([1.0, 3.0]),
        curl_operator=np.array([[1.0, -1.0]]),
        time_intervals=np.array([1.0, 2.0]),
        interval_average_didt=np.array([3.0, 4.0]),
        current_initial=1.0,
        current_final=5.0,
    )

    assert diagnostics["source_endpoint_balance_residual"] == 1.0
    assert diagnostics["dc_current_conservation_residual"] == np.sqrt(5.0)
    assert diagnostics["initial_curl_residual"] == 2.0
    assert diagnostics["waveform_integral_residual"] == 7.0
