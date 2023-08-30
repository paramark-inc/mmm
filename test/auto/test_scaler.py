import numpy as np
import unittest

from mmm.data.serializable_scaler import SerializableScaler


class SerializableScalerTest(unittest.TestCase):
    def test_serializable_scaler_to_dict(self):
        scaler = SerializableScaler()
        data = np.array([1, 2, 3, 4])
        scaler.fit(data)

        self.assertDictEqual(scaler.to_dict(), {"divide_by": 2.5, "multiply_by": 1.0})

    def test_serializable_scaler_from_dict(self):
        scaler = SerializableScaler()
        scaler.from_dict(
            {
                "multiply_by": [2, 1],
                "divide_by": [1, 2],
            }
        )

        data = np.ones(2)
        transformed = scaler.transform(data)
        expected = np.array([2, 0.5])

        np.testing.assert_array_equal(transformed, expected)
