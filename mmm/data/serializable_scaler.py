# note that JAX imports are weird: jaxlib.xla_extension
# is not available until jax.numpy has been imported
import jaxlib
import jax.numpy as jnp

from impl.lightweight_mmm.lightweight_mmm.preprocessing import CustomScaler, NotFittedScalerError


class SerializableScaler(CustomScaler):
    """
    Modified version of lightweight_mmm's CustomScaler:
    * always use a divide operation that ignores zero values
    * add methods to allow the scaler to be serialized/deserialized by
      generating dicts with regular floats and lists, not JAX arrays
    """

    def __init__(self, *args, **kwargs):
        return super(SerializableScaler, self).__init__(
            divide_operation=SerializableScaler._robust_scaling_divide_operation, *args, **kwargs
        )

    @staticmethod
    def _robust_scaling_divide_operation(x):
        """
        Scaling divide operation that is robust to zero values (unlike jnp.mean).

        :param x: array of values
        :return: sum / count_of_positive_values
        """
        # special case when all rows are zero so that we don't divide by zero.  We could use
        # any value here, since the numerators will all be zero, so we use 1.
        n_elements_gt_zero = (x > 0).sum()
        return jnp.where(n_elements_gt_zero > 0, x.sum() / n_elements_gt_zero, 1)

    def to_dict(self):
        if not hasattr(self, "divide_by") or not hasattr(self, "multiply_by"):
            raise NotFittedScalerError("Can't serialize a scaler before it has been fit()")

        multiply_by = None
        if isinstance(self.multiply_by, jaxlib.xla_extension.ArrayImpl):
            # tolist() works for all jax "arrays" even if they are zero-dimension (i.e. scalar)
            multiply_by = self.multiply_by.tolist()
        elif isinstance(self.multiply_by, list):
            # for scalers created via from_dict, multiply/divide might be a standard list (not JAX)
            multiply_by = self.multiply_by
        else:
            # this is likely a numpy.float64
            multiply_by = float(self.multiply_by)

        divide_by = None
        if isinstance(self.divide_by, jaxlib.xla_extension.ArrayImpl):
            divide_by = self.divide_by.tolist()
        elif isinstance(self.divide_by, list):
            divide_by = self.divide_by
        else:
            divide_by = float(self.divide_by)

        return {
            "multiply_by": multiply_by,
            "divide_by": divide_by,
        }

    def from_dict(self, input_dict):
        # note that we use init from the parent class, not from this class:
        # in the parent constructor, the presence of divide_operation would
        # override the value we set for divide_by
        return super(SerializableScaler, self).__init__(**input_dict)
