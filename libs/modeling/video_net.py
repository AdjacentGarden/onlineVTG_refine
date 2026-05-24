from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import (
    sinusoid_encoding,
    MaskedConv1D,
    LayerNorm,
    masked_max_pool1d,
    LayerScale,
    FFN,
)


backbones = dict()


def register_video_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator


class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask.

    这里保留你项目原来的 MaskedMHA 结构。
    它主要用于 OnlineTransformerEncoder 内部的当前层 self-attention / history attention。

    注意：
    这个模块没有严格的窗口内部 causal mask。
    如果你的 online 定义是 chunk-level online，也就是 short_window 内部可以整体处理，
    那么这个设计可以接受。
    如果你的 online 定义是 frame-level strict online，
    还需要额外加窗口内部 causal mask。
    """
    def __init__(
        self,
        embd_dim,
        q_dim=None,
        kv_dim=None,
        out_dim=None,
        n_heads=4,
        window_size=0,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
    ):
        super(MaskedMHA, self).__init__()

        assert embd_dim % n_heads == 0

        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / np.sqrt(np.sqrt(self.n_channels))
        self.out_dim = out_dim

        self.query = MaskedConv1D(q_dim, embd_dim, kernel_size=1, stride=1, padding=0)
        self.key = MaskedConv1D(kv_dim, embd_dim, kernel_size=1, stride=1, padding=0)
        self.value = MaskedConv1D(kv_dim, embd_dim, kernel_size=1, stride=1, padding=0)
        self.proj = MaskedConv1D(embd_dim, out_dim, kernel_size=1, stride=1, padding=0)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        assert window_size == 0 or window_size % 2 == 1
        self.window_size = window_size
        self.stride = window_size // 2

        self.l_mask = None
        self.r_mask = None

    def forward(self, x, mask, history=None, history_mask=None):
        """
        x:
            当前窗口特征，shape = (B, C, T)

        mask:
            当前窗口 mask，shape = (B, 1, T)

        history:
            历史特征，shape = (B, C, T_history)

        history_mask:
            历史 mask，shape = (B, 1, T_history)
        """
        bs = x.size(0)
        c = self.embd_dim
        h = self.n_heads
        d = self.n_channels

        q, _ = self.query(x, mask)
        k, _ = self.key(x, mask)
        v, _ = self.value(x, mask)

        if history is not None:
            history_k, _ = self.key(history, history_mask)
            history_v, _ = self.value(history, history_mask)
        else:
            history_k = None
            history_v = None

        q = q.view(bs, h, d, -1).transpose(2, 3).contiguous()
        k = k.view(bs, h, d, -1)
        v = v.view(bs, h, d, -1).transpose(2, 3).contiguous()

        if history is not None:
            history_k = history_k.view(bs, h, d, -1)
            history_v = history_v.view(bs, h, d, -1).transpose(2, 3).contiguous()

            k = torch.cat([k, history_k], dim=-1)
            v = torch.cat([v, history_v], dim=2)
            mask = torch.cat([mask, history_mask], dim=-1)

        attn = (q * self.scale) @ (k * self.scale)

        attn = attn.masked_fill(
            mask=torch.logical_not(mask[:, :, None, :]),
            value=-1e9,
        )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        q = attn @ v

        q = q.transpose(2, 3).contiguous().view(bs, c, -1)

        out, _ = self.proj(q, None)
        out = self.proj_drop(out)

        return out


class MaskedCrossMHA(nn.Module):
    """
    使用项目已有 MaskedConv1D 实现的 masked cross-attention。

    Query 来自 current feature。
    Key / Value 来自 memory feature。

    用途：
    1. current attends to same-level memory；
    2. current attends to upper-level memory。

    与 MaskedMHA 的区别：
    - MaskedMHA 原来会把 current 自身和 history 拼接为 key/value；
    - 这里是真正的 cross-attention，只用 memory 作为 key/value；
    - 更适合做 memory fusion。
    """
    def __init__(
        self,
        embd_dim,
        q_dim=None,
        kv_dim=None,
        out_dim=None,
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
    ):
        super().__init__()

        assert embd_dim % n_heads == 0

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.embd_dim = embd_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / np.sqrt(np.sqrt(self.n_channels))

        # 全部使用项目已有的 MaskedConv1D
        self.query = MaskedConv1D(q_dim, embd_dim, kernel_size=1, stride=1, padding=0)
        self.key = MaskedConv1D(kv_dim, embd_dim, kernel_size=1, stride=1, padding=0)
        self.value = MaskedConv1D(kv_dim, embd_dim, kernel_size=1, stride=1, padding=0)
        self.proj = MaskedConv1D(embd_dim, out_dim, kernel_size=1, stride=1, padding=0)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

    def forward(self, q_x, q_mask, kv_x, kv_mask):
        """
        q_x:
            当前特征，shape = (B, C, T_q)

        q_mask:
            当前 mask，shape = (B, 1, T_q)

        kv_x:
            memory 特征，shape = (B, C, T_kv)

        kv_mask:
            memory mask，shape = (B, 1, T_kv)
        """
        B = q_x.size(0)
        H = self.n_heads
        D = self.n_channels
        C = self.embd_dim

        q, _ = self.query(q_x, q_mask)
        k, _ = self.key(kv_x, kv_mask)
        v, _ = self.value(kv_x, kv_mask)

        q = q.view(B, H, D, -1).transpose(2, 3).contiguous()
        k = k.view(B, H, D, -1)
        v = v.view(B, H, D, -1).transpose(2, 3).contiguous()

        attn = (q * self.scale) @ (k * self.scale)

        attn = attn.masked_fill(
            mask=torch.logical_not(kv_mask[:, :, None, :]),
            value=-1e9,
        )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        out, _ = self.proj(out, q_mask)
        out = self.proj_drop(out)
        out = out * q_mask.to(out.dtype)

        return out


class OnlineTransformerEncoder(nn.Module):
    """
    Online Transformer Encoder.

    结构：
    1. MaskedMHA
    2. residual
    3. FFN
    4. residual
    """
    def __init__(
        self,
        embd_dim,
        n_heads=4,
        window_size=0,
        expansion=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
    ):
        super(OnlineTransformerEncoder, self).__init__()

        self.attn = MaskedMHA(
            embd_dim=embd_dim,
            n_heads=n_heads,
            window_size=window_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
        )

        self.ln_attn = LayerNorm(embd_dim)
        self.drop_path_attn = LayerScale(embd_dim, path_pdrop)

        self.ffn = FFN(embd_dim, expansion, proj_pdrop)
        self.ln_ffn = LayerNorm(embd_dim)
        self.drop_path_ffn = LayerScale(embd_dim, path_pdrop)

    def forward(self, x, mask, history=None, history_mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)

        x = x * mask.to(x.dtype)

        if history is None:
            h = self.attn(
                self.ln_attn(x),
                mask,
            ) * mask.to(x.dtype)
        else:
            history = history * history_mask.to(history.dtype)

            h = self.attn(
                self.ln_attn(x),
                mask,
                self.ln_attn(history),
                history_mask,
            ) * mask.to(x.dtype)

        x = x * mask.to(x.dtype) + self.drop_path_attn(h)

        h = self.ffn(self.ln_ffn(x)) * mask.to(x.dtype)
        x = x + self.drop_path_ffn(h)

        return x, mask


class EventMemory:
    """
    每个尺度维护一个 memory。

    self.history[scale]:
        第 scale 层的历史特征。

    self.history_mask[scale]:
        第 scale 层的历史 mask。

    memory_size[scale]:
        第 scale 层最多保留多少历史 token。
    """
    def __init__(self, n_scales, memory_size, threshold=0.95):
        self.n_scales = n_scales
        self.memory_size = memory_size
        self.history = []
        self.history_mask = []
        self.threshold = threshold

    def read(self, scale):
        if len(self.history) <= scale:
            return None, None

        return self.history[scale], self.history_mask[scale]

    def update(self, scale, x, mask):
        if len(self.history) <= scale:
            self.history.append(x)
            self.history_mask.append(mask)
        else:
            self.history[scale] = torch.cat(
                [self.history[scale], x],
                dim=-1,
            )
            self.history_mask[scale] = torch.cat(
                [self.history_mask[scale], mask],
                dim=-1,
            )

        if self.history[scale].size(-1) > self.memory_size[scale]:
            self.history[scale] = self.history[scale][
                ...,
                -self.memory_size[scale]:
            ]
            self.history_mask[scale] = self.history_mask[scale][
                ...,
                -self.memory_size[scale]:
            ]

    def adaptive_merge(self, T):
        """
        原始代码中已有的 adaptive merge。
        当前 OnlineVideoTransformer.forward() 默认没有调用。
        """
        B, dim, length = T.shape
        device = T.device

        t1 = T[:, :, :-1]
        t2 = T[:, :, 1:]

        similarities = F.cosine_similarity(t1, t2, dim=1)
        max_sim_values, max_sim_indices = torch.max(similarities, dim=1)
        should_merge_mask = max_sim_values > self.threshold

        pooled_vectors = (t1 + t2) / 2.0
        source_for_merge = torch.cat([T, pooled_vectors], dim=2)

        k = max_sim_indices.unsqueeze(1)
        base_indices = torch.arange(length - 1, device=device).expand(B, -1)

        indices_lt_k = base_indices
        indices_eq_k = base_indices + length
        indices_gt_k = base_indices + 1

        merge_gather_indices = torch.where(
            base_indices < k,
            indices_lt_k,
            torch.where(
                base_indices == k,
                indices_eq_k,
                indices_gt_k,
            ),
        )

        merge_gather_indices = merge_gather_indices.unsqueeze(1).expand(
            B,
            dim,
            length - 1,
        )

        merged_result = torch.gather(
            source_for_merge,
            2,
            merge_gather_indices,
        )

        discarded_result = T[:, :, 1:]

        final_mask = should_merge_mask.view(B, 1, 1).expand_as(merged_result)

        final_output = torch.where(
            final_mask,
            merged_result,
            discarded_result,
        )

        return final_output

    def clear(self):
        self.history.clear()
        self.history_mask.clear()


class CausalHierarchicalMemoryFusion(nn.Module):
    """
    Causal Hierarchical Memory Fusion, CHMF.

    当前层 current feature 同时融合：
    1. 当前层历史 memory，same-level memory；
    2. 上一层历史 memory，upper-level memory，也就是更粗尺度 memory。

    这里尽量使用项目已有模块：
    - cross attention 内部使用 MaskedConv1D；
    - gate 里的 1x1 conv 使用 MaskedConv1D；
    - output projection 使用 MaskedConv1D；
    - norm 使用项目里的 LayerNorm。

    融合方式：
    current -> same-level memory cross attention -> same_enhanced
    current -> upper-level memory cross attention -> upper_enhanced
    current / same_enhanced / upper_enhanced -> gate fusion

    因果性保证：
    - same_memory 来自当前层之前已经写入的 memory；
    - upper_memory 来自上一层之前已经写入的 memory；
    - 不使用当前窗口刚 pool 出来的 upper feature；
    - 因此不会让当前 token 看到未来 token 聚合后的粗尺度信息。
    """
    def __init__(
        self,
        embd_dim,
        n_levels,
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        init_res_gate=-4.0,
        use_same_memory=True,
        use_upper_memory=True,
    ):
        super().__init__()

        self.embd_dim = embd_dim
        self.n_levels = n_levels
        self.use_same_memory = use_same_memory
        self.use_upper_memory = use_upper_memory

        self.same_attn = nn.ModuleList([
            MaskedCrossMHA(
                embd_dim=embd_dim,
                q_dim=embd_dim,
                kv_dim=embd_dim,
                out_dim=embd_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
            for _ in range(n_levels)
        ])

        self.upper_attn = nn.ModuleList([
            MaskedCrossMHA(
                embd_dim=embd_dim,
                q_dim=embd_dim,
                kv_dim=embd_dim,
                out_dim=embd_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
            for _ in range(n_levels)
        ])

        self.norm_current_for_same = nn.ModuleList([
            LayerNorm(embd_dim)
            for _ in range(n_levels)
        ])

        self.norm_same_memory = nn.ModuleList([
            LayerNorm(embd_dim)
            for _ in range(n_levels)
        ])

        self.norm_current_for_upper = nn.ModuleList([
            LayerNorm(embd_dim)
            for _ in range(n_levels)
        ])

        self.norm_upper_memory = nn.ModuleList([
            LayerNorm(embd_dim)
            for _ in range(n_levels)
        ])

        # gate 分支不使用 nn.Conv1d，改用 MaskedConv1D
        self.gate_reduce = nn.ModuleList([
            MaskedConv1D(
                embd_dim * 3,
                embd_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for _ in range(n_levels)
        ])

        self.gate_out = nn.ModuleList([
            MaskedConv1D(
                embd_dim,
                3,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for _ in range(n_levels)
        ])

        self.out_norm = nn.ModuleList([
            LayerNorm(embd_dim)
            for _ in range(n_levels)
        ])

        self.out_proj = nn.ModuleList([
            MaskedConv1D(
                embd_dim,
                embd_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for _ in range(n_levels)
        ])

        # 残差门控。
        # 这里 nn.Parameter 是必要的参数容器，不是计算层。
        self.res_gate = nn.Parameter(
            torch.full(
                size=(n_levels,),
                fill_value=float(init_res_gate),
            )
        )

    def _has_valid_memory(self, memory, memory_mask):
        if memory is None or memory_mask is None:
            return False

        if memory.size(-1) == 0:
            return False

        if memory_mask.size(-1) == 0:
            return False

        if not torch.any(memory_mask):
            return False

        return True

    def _compute_gate(
        self,
        gate_input,
        gate_mask,
        scale,
        has_same,
        has_upper,
    ):
        """
        用项目里的 MaskedConv1D 计算三路 gate。

        输出 gate_weight:
            shape = (B, 3, T)
        """
        gate_hidden, _ = self.gate_reduce[scale](
            gate_input,
            gate_mask,
        )
        gate_hidden = F.relu(gate_hidden, inplace=True)

        gate_logits, _ = self.gate_out[scale](
            gate_hidden,
            gate_mask,
        )

        if not has_same:
            gate_logits[:, 1:2, :] = -1e4

        if not has_upper:
            gate_logits[:, 2:3, :] = -1e4

        gate_weight = F.softmax(gate_logits, dim=1)

        return gate_weight

    def forward(
        self,
        current,
        current_mask,
        same_memory=None,
        same_memory_mask=None,
        upper_memory=None,
        upper_memory_mask=None,
        scale=0,
    ):
        """
        current:
            当前层当前窗口特征，shape = (B, C, T)

        current_mask:
            当前层当前窗口 mask，shape = (B, 1, T)

        same_memory:
            当前层历史 memory，shape = (B, C, T_same)

        same_memory_mask:
            当前层历史 memory mask，shape = (B, 1, T_same)

        upper_memory:
            上一层历史 memory，shape = (B, C, T_upper)

        upper_memory_mask:
            上一层历史 memory mask，shape = (B, 1, T_upper)
        """
        same_enhanced = torch.zeros_like(current)
        upper_enhanced = torch.zeros_like(current)

        has_same = (
            self.use_same_memory
            and self._has_valid_memory(same_memory, same_memory_mask)
        )

        has_upper = (
            self.use_upper_memory
            and self._has_valid_memory(upper_memory, upper_memory_mask)
        )

        if has_same:
            same_enhanced = self.same_attn[scale](
                q_x=self.norm_current_for_same[scale](current),
                q_mask=current_mask,
                kv_x=self.norm_same_memory[scale](same_memory),
                kv_mask=same_memory_mask,
            )

        if has_upper:
            upper_enhanced = self.upper_attn[scale](
                q_x=self.norm_current_for_upper[scale](current),
                q_mask=current_mask,
                kv_x=self.norm_upper_memory[scale](upper_memory),
                kv_mask=upper_memory_mask,
            )

        gate_input = torch.cat(
            [
                current,
                same_enhanced,
                upper_enhanced,
            ],
            dim=1,
        )

        gate_weight = self._compute_gate(
            gate_input=gate_input,
            gate_mask=current_mask,
            scale=scale,
            has_same=has_same,
            has_upper=has_upper,
        )

        w_current = gate_weight[:, 0:1, :]
        w_same = gate_weight[:, 1:2, :]
        w_upper = gate_weight[:, 2:3, :]

        fused = (
            w_current * current
            + w_same * same_enhanced
            + w_upper * upper_enhanced
        )

        fused = self.out_norm[scale](fused)

        fused, _ = self.out_proj[scale](
            fused,
            current_mask,
        )

        alpha = torch.sigmoid(self.res_gate[scale])

        out = current + alpha * fused
        out = out * current_mask.to(out.dtype)

        return out, current_mask


@register_video_net("online_transformer")
class OnlineVideoTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        embd_dim,
        max_seq_len,
        n_heads,
        stride=1,
        n_convs=2,
        n_encoder_layers=8,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_ref_pe=False,
        short_window_size=8,
        memory_size=[8, 8, 8, 8, 8, 8, 8, 8],

        # 新增参数：是否启用 CHMF
        use_hierarchical_memory_fusion=True,

        # 新增参数：CHMF 的残差门控初始化
        hierarchical_memory_init_gate=-4.0,

        # 新增参数：是否使用当前层历史 memory
        use_same_level_memory=True,

        # 新增参数：是否使用上一层历史 memory
        use_upper_level_memory=True,

        **kargs,
    ):
        super().__init__()

        assert stride & (stride - 1) == 0
        assert n_convs >= int(math.log2(stride))

        self.max_seq_len = max_seq_len
        self.stride = stride

        self.embd_fc = MaskedConv1D(
            in_dim,
            embd_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()

        current_stride = stride

        for _ in range(n_convs):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim,
                    embd_dim,
                    kernel_size=5 if current_stride > 1 else 3,
                    stride=2 if current_stride > 1 else 1,
                    padding=2 if current_stride > 1 else 1,
                    bias=False,
                )
            )

            self.embd_norms.append(LayerNorm(embd_dim))

            current_stride = max(current_stride // 2, 1)

        if use_abs_pe:
            self.pe_type = "abs"
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer("pe", pe, persistent=False)
        elif use_ref_pe:
            self.pe_type = "ref"
            self.max_pe = 256
            self.pe = nn.Parameter(
                torch.randn(embd_dim, self.max_pe) / embd_dim ** 0.5
            )
        else:
            self.pe_type = None
            self.pe = None

        self.branch = nn.ModuleList()

        for _ in range(n_encoder_layers):
            self.branch.append(
                OnlineTransformerEncoder(
                    embd_dim,
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.short_window_size = short_window_size

        self.memory = EventMemory(
            n_scales=n_encoder_layers,
            memory_size=memory_size,
        )

        self.use_hierarchical_memory_fusion = use_hierarchical_memory_fusion

        self.hierarchical_memory_fusion = CausalHierarchicalMemoryFusion(
            embd_dim=embd_dim,
            n_levels=n_encoder_layers,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            init_res_gate=hierarchical_memory_init_gate,
            use_same_memory=use_same_level_memory,
            use_upper_memory=use_upper_level_memory,
        )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        """
        兼容 MaskedConv1D 的初始化。

        由于不同项目里 MaskedConv1D 可能把真实 conv 命名为：
        - conv
        - masked_conv
        - proj
        等等。

        所以这里做一个保守初始化：
        如果模块自身有 bias，就置零；
        如果模块内部有 conv 且 conv 有 bias，也置零。
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if hasattr(module, "conv"):
            conv = getattr(module, "conv")
            if hasattr(conv, "bias") and conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def apply_pe(self, x, mask):
        if self.pe_type == "ref":
            pe_all = []
            b, d, n = x.shape

            for i in range(b):
                t = mask[i].sum().item()

                pe = F.interpolate(
                    self.pe[None],
                    size=t,
                    mode="nearest",
                )

                if t < n:
                    pe = torch.cat(
                        [
                            pe,
                            torch.zeros(
                                (1, d, n - t),
                                dtype=pe.dtype,
                                device=pe.device,
                            ),
                        ],
                        dim=-1,
                    )

                pe_all.append(pe)

            pe_all = torch.cat(pe_all, dim=0)
            x = x + pe_all

        else:
            _, _, t = x.size()
            pe = self.pe.to(x.dtype)

            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None],
                        size=t,
                        mode="linear",
                        align_corners=True,
                    )[0]

            x = x + pe[..., :t] * mask.to(x.dtype)

        return x

    def _append_fpn_output(
        self,
        fpn,
        fpn_masks,
        scale,
        current,
        current_mask,
        bs,
        nw,
    ):
        current_out = current.view(
            bs,
            nw,
            current.size(1),
            -1,
        )

        current_mask_out = current_mask.view(
            bs,
            nw,
            1,
            -1,
        )

        if len(fpn) <= scale:
            fpn.append(current_out)
            fpn_masks.append(current_mask_out)
        else:
            fpn[scale] = torch.cat(
                [fpn[scale], current_out],
                dim=-1,
            )

            fpn_masks[scale] = torch.cat(
                [fpn_masks[scale], current_mask_out],
                dim=-1,
            )

    def forward(self, x, mask):
        """
        x:
            shape = (bs, nw, c, vlen)

        mask:
            shape = (bs, nw, vlen)

        return:
            tuple(fpn), tuple(fpn_masks)

        修改后的核心流程：

        对于每个 short window：
            对于每个 scale：
                1. 读取当前层历史 memory，same_history；
                2. 当前层 branch encoder；
                3. 读取上一层历史 memory，upper_history；
                4. current 分别和 same_history / upper_history 做 cross-attention；
                5. current / same_enhanced / upper_enhanced 三路 gate 融合；
                6. 更新当前层 memory；
                7. 保存当前层 FPN 输出；
                8. 满足时间边界时，bottom-up 下采样到下一层。

        因果性关键：
            upper_history 是上一层之前已经得到的 memory，
            不是当前窗口刚 pool 出来的 upper feature。
            所以不会把当前窗口未来 token 的信息泄露给当前 token。
        """
        bs, nw, _, vlen = x.shape

        x = x.view(bs * nw, -1, vlen)
        mask = mask.view(bs * nw, vlen)

        self.memory.clear()

        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        x, _ = self.embd_fc(x, mask)

        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        if self.pe is not None:
            x = self.apply_pe(x, mask)

        fpn = []
        fpn_masks = []

        total_len = x.size(-1)

        for i in range(0, total_len, self.short_window_size):
            current = x[..., i: i + self.short_window_size]
            current_mask = mask[..., i: i + self.short_window_size]

            for scale in range(len(self.branch)):
                same_history, same_history_mask = self.memory.read(scale)

                # ! new part
                current, current_mask = self.branch[scale](
                    current,
                    current_mask,
                    None,
                    None
                )

                # 2. 读取上一层，也就是更粗尺度历史 memory。
                # 注意：
                # 这里读取的是已经存在的 memory[scale + 1]，
                # 不是当前窗口刚由 current pool 出来的粗尺度特征。
                upper_history = None
                upper_history_mask = None

                if scale + 1 < len(self.branch):
                    upper_history, upper_history_mask = self.memory.read(
                        scale + 1
                    )

                # 3. CHMF 融合：
                # current 和当前层历史 memory 融合；
                # current 和上一层历史 memory 融合；
                # 最后三路 gate 融合。
                if self.use_hierarchical_memory_fusion:
                    current, current_mask = self.hierarchical_memory_fusion(
                        current=current,
                        current_mask=current_mask,
                        same_memory=same_history,
                        same_memory_mask=same_history_mask,
                        upper_memory=upper_history,
                        upper_memory_mask=upper_history_mask,
                        scale=scale,
                    )

                # 4. 融合完成后，再更新当前层 memory。
                # 不能先 update 再 fusion，否则当前窗口特征可能提前进入 memory，
                # 导致当前窗口内部信息回流。
                self.memory.update(
                    scale,
                    current,
                    current_mask,
                )

                # 5. 保存当前尺度 FPN 输出。
                self._append_fpn_output(
                    fpn=fpn,
                    fpn_masks=fpn_masks,
                    scale=scale,
                    current=current,
                    current_mask=current_mask,
                    bs=bs,
                    nw=nw,
                )

                if scale + 1 == len(self.branch):
                    break

                if (i + self.short_window_size) % (2 ** (scale + 1)) != 0:
                    break

                if current.size(-1) % 2 != 0:
                    if same_history is not None and same_history_mask is not None:
                        current = torch.cat(
                            [same_history[..., -1:], current],
                            dim=-1,
                        )

                        current_mask = torch.cat(
                            [same_history_mask[..., -1:], current_mask],
                            dim=-1,
                        )
                    else:
                        break

                current, current_mask = masked_max_pool1d(
                    current,
                    current_mask,
                    kernel_size=2,
                    stride=2,
                )

        return tuple(fpn), tuple(fpn_masks)


def make_video_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop("name")](**opt)