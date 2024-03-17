import pytest
import torch
import torch_npu

from ascendspeed.op_builder import FusionAttentionOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNPUFusionAttention(DistributedTest):
    world_size = 1

    def supported_op_exec(self, query, key, value):
        scale = 0.08838
        return torch_npu.npu_fusion_attention(query, key, value, head_num=32, input_layout="BSH", scale=scale)

    def custom_op_exec(self, query, key, value):
        scale = 0.08838
        ascendspeed_ops = FusionAttentionOpBuilder().load()
        return ascendspeed_ops.npu_fusion_attention(query, key, value, 32, "BSH", None, None, None, scale, 1.,
                                                    2147483647, 2147483647, 0, None, None, None, 0, True, False)

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_flash_attention(self):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(q_npu, k_npu, v_npu)
        attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = self.custom_op_exec(q_npu, k_npu,
                                                                                                           v_npu)
        assert torch.allclose(output[0], attention_score, rtol=0.001, atol=0.001)
