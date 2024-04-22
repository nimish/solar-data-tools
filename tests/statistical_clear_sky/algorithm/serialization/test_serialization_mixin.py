import numpy as np
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting


def test_serialization(tmp_path):
    file_name = tmp_path / "state_data.json"
    power_signals_d = np.array(
        [
            [3.65099996e-01, 0.00000000e00, 0.00000000e00, 2.59570003e00],
            [6.21100008e-01, 0.00000000e00, 0.00000000e00, 2.67740011e00],
            [8.12500000e-01, 0.00000000e00, 0.00000000e00, 2.72729993e00],
            [9.00399983e-01, 0.00000000e00, 0.00000000e00, 2.77419996e00],
        ]
    )
    rank_k = 4

    original_iterative_fitting = IterativeFitting(power_signals_d, rank_k=rank_k)

    original_iterative_fitting.save_instance(file_name)

    deserialized_iterative_fitting = IterativeFitting.load_instance(file_name)

    np.testing.assert_array_equal(
        deserialized_iterative_fitting._power_signals_d,
        original_iterative_fitting._power_signals_d,
    )
