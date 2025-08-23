import jax.numpy as jnp
from image_converter_utils import jxl_xyb_to_srgb, srgb_to_jxl_xyb, dct_to_xyb, xyb_to_dct, upscale
import equinox as eqx
import jax

class OptimizerValues(eqx.Module):
    values: list[jax.Array]

    def __init__(self, shape, layers):
        self.values = [jnp.zeros((shape[0]//(2**i), shape[1]//(2**i), shape[2])) for i in range(layers)]

    def convert_to_rgb(self):
        pass

    def convert_to_xyb(self):
        pass

    def convert_to_xyb_dct(self):
        pass

    def combine_to_rgb(self):
        xyb_upscaled = jnp.array([upscale(image, 2**i) for i, image in enumerate(self.convert_to_xyb())])
        return jxl_xyb_to_srgb(jnp.sum(xyb_upscaled, axis=0))

class RGBOptimizerValues(OptimizerValues):
    def convert_to_rgb(self):
        return self.values

    def convert_to_xyb(self):
        return [srgb_to_jxl_xyb(val) for val in self.values]

    def convert_to_xyb_dct(self):
        return [xyb_to_dct(val) for val in self.convert_to_xyb()]


class XYBOptimizerValues(OptimizerValues):
    def convert_to_xyb(self):
        return self.values

    def convert_to_rgb(self):
        return [jxl_xyb_to_srgb(val) for val in self.values]

    def convert_to_xyb_dct(self):
        return [xyb_to_dct(val) for val in self.values]

class XYBDCTOptimizerValues(OptimizerValues):

    def __init__(self, shape, layers):
        super().__init__(shape, layers)
        self.values =[jnp.zeros((shape[0]//(2**i)//8, shape[1]//(2**i)//8, shape[2], 8, 8)) for i in range(layers)]

    def convert_to_xyb(self):
        return [dct_to_xyb(val) for val in self.values]

    def convert_to_xyb_dct(self):
        return self.values

    def convert_to_rgb(self):
        return [jxl_xyb_to_srgb(val) for val in self.convert_to_xyb()]