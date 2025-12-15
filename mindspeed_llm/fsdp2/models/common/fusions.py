import torch
import torch_npu


def fused_rmsnorm_forward(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def fused_mlp_forward(self, x):
    gate_up_w = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
    gate_up_output = torch.matmul(x, gate_up_w.t())

    swiglu_output = torch_npu.npu_swiglu(gate_up_output, dim=-1)
    down_proj = self.down_proj(swiglu_output)
    return down_proj
