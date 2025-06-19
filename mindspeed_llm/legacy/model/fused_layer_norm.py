import inspect

from apex.normalization.fused_layer_norm import fused_layer_norm_affine


def fused_layer_norm_forward(self, input):

    weight = self.weight + 1 if self.apply_layernorm_1p else self.weight
    
    if fused_layer_norm_affine is None:
        raise AssertionError("fused_layer_norm_affine is not available, please install apex")
    return fused_layer_norm_affine(input, weight, self.bias, self.normalized_shape, eps=self.eps)