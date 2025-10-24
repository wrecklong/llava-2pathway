import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from einops import rearrange

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10
)
from transformers.activations import ACT2FN

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis
    from flash_attn import flash_attn_varlen_func
from torch.nn import functional as F

class ScaleDotProductCrossAttention(nn.Module):
    
    def __init__(self, layer_number, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.layer_number = layer_number
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, attn_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        # (N,...,L,E)

        if attn_mask is not None:
            attn_mask = attn_mask[:,None,:,:].repeat(1, q.shape[1], 1, 1)

        # attention mask, True means it will take part in attention B H s_q s_k
        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0.0

        if q.device.type == "cuda" and attn_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        
        # debug only, calculate the FLOPs for cross-attn
        ##################
        # attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(128) # hardcode
        # if attn_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attn_mask[:, :, :, : k.shape[-2]]
        #     attn_weights = attn_weights + causal_mask

        # # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # o = torch.matmul(attn_weights, v)
        ###################

        o = nn.functional.scaled_dot_product_attention(q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.softmax_scale)
        
        # B Head L D -> L B (Head D)
        o = rearrange(o, 'B Head L D -> B L (Head D)').contiguous()
        
        return o

class FlashAttnCrossAttention(nn.Module):
    
    def __init__(self, layer_number, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.layer_number = layer_number
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def _get_unpad_data(self, attention_mask: torch.Tensor):
        """
        Retrieves indexing data required to repad unpadded (ragged) tensors.

        Arguments:
            attention_mask (`torch.Tensor`):
                Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

        Return:
            indices (`torch.Tensor`):
                The indices of non-masked tokens from the flattened input sequence.
            cu_seqlens (`torch.Tensor`):
                The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
            max_seqlen_in_batch (`int`):
                Maximum sequence length in batch.
        """
        seqlens_in_batch = attention_mask[:, 0, :].sum(dim=-1, dtype=torch.int32) # attn mask are the same for the query dimension, pick the first query
        indices = torch.nonzero(attention_mask[:, 0, :].flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            seqlens_in_batch
        )
    def unpad_q(self, q_layer):
        # no need to unpad, just flatten
        
        batch_size, q_seq_len, num_key_value_heads, head_dim = q_layer.shape
        cu_seqlens_q = torch.tensor([q_seq_len] * batch_size, dtype=torch.int32, device=q_layer.device)
        cu_seqlens_q = nn.functional.pad(torch.cumsum(cu_seqlens_q, dim=0, dtype=torch.int32), (1, 0))    
        q_layer = q_layer.reshape(batch_size * q_seq_len, num_key_value_heads, head_dim)
    
        return (
            q_layer,
            cu_seqlens_q,
            q_seq_len)
    def unpad_kv(self, key_layer, value_layer, attn_mask):
        
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k, split_size = self._get_unpad_data(attn_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        return (
            key_layer,
            value_layer,
            indices_k,
            cu_seqlens_k,
            max_seqlen_in_batch_k,
            split_size)
    
    def forward(self, q, k, v, attn_mask=None):
        """
        Implements the multihead softmax attention with flash attention varlen api.
        Unpad the kv sequence
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        # (N,...,L,E)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # NOTE: don't know if it's necessary
        if q.device.type == "cuda" and attn_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        # batch_size = q.shape[0]
        # first unpad the q and kv, get cu_seq_len and indices
        batch_size, q_seq_len, head_num, head_dim = q.shape
        q, cu_seq_lens_q, max_seqlen_in_batch_q = self.unpad_q(q)
        k, v, indices_kv, cu_seq_lens_kv, max_seqlen_in_batch_kv, split_size = self.unpad_kv(k, v, attn_mask)
        
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_kv,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_kv,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
            # **flash_kwargs
        )

        return attn_output.reshape(batch_size, q_seq_len, head_num, head_dim).flatten(2, 3).contiguous()

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config=None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class Qwen3Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # if layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
        #         "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # self.is_causal = True
        # self.attention_dropout = config.attention_dropout

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )
        
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3FlashAttention2(Qwen3Attention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            # logger.warning_once(
            #     f"The input hidden states seems to be silently casted in float32, this might be related to"
            #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            #     f" {target_dtype}."
            # )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        q_len = query_states.shape[2]
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )


        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3HybridFlashAttention2(Qwen3FlashAttention2):
    """
    Qwen3 flash attention module, following Qwen3 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, 
                 is_hyper_enabled, 
                 gating_type, 
                 cross_attn_implementation, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        
        self.is_hyper_enabled = is_hyper_enabled
        if self.is_hyper_enabled:
            self.gating_type = gating_type
            self.cross_attention_implementation = cross_attn_implementation
            # self.cross_attn_kv_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim * 2, bias=False)

            # with torch.no_grad():
            #     # 将 cross_attn_kv_proj.weight 拆成两半，分别绑定到 k_proj.weight 和 v_proj.weight
            #     self.cross_attn_kv_proj.weight[:self.num_key_value_heads * self.head_dim, :] = self.k_proj.weight
            #     self.cross_attn_kv_proj.weight[self.num_key_value_heads * self.head_dim:, :] = self.v_proj.weight
            
            if gating_type.startswith("whole-dynamic"):
                if "tanh" in gating_type:
                    self.cross_attn_gate_proj = nn.Sequential(
                        nn.Linear(self.num_heads * self.head_dim , 1),
                        nn.Tanh()
                    )
                else:
                    self.cross_attn_gate_proj = nn.Sequential(
                        nn.Linear(self.num_heads * self.head_dim, 1),
                    )
                    
                if gating_type.endswith("warmup"):
                    self.cross_attn_warm_up_gate = torch.nn.Parameter(torch.zeros(1))
                
            if "flashattn" in self.cross_attention_implementation:
                self.cross_attn_core_attention = FlashAttnCrossAttention(layer_number=-1, attention_dropout=self.attention_dropout)
            else:
                self.cross_attn_core_attention = ScaleDotProductCrossAttention(layer_number=-1, attention_dropout=self.attention_dropout)
    
    
    def all2media_cross_attn(self, 
                              text_state, 
                              text_query,
                              vision_features,
                              text2vision_cross_attn_mask=None,
                              all_text_mask=None):
        '''
        text_query: [s b h d]
        text_state: s b d
        vision_features: [num_vis, b,  d]
        '''        

        if vision_features is None or (self.is_hyper_enabled == False):
            return text_state
        
        L_c, B_c = text_state.shape[:2]
        D_head = self.head_dim
        
        if "whole-dynamic" in self.gating_type:
            gate_value = self.cross_attn_gate_proj(text_state) # n, bs, head_D
            if "warmup" in self.gating_type:
                gate_value = gate_value * self.cross_attn_warm_up_gate

        vision_features = vision_features.contiguous()
        vision_features = self.cross_attn_kv_proj(vision_features)        
        text_query = rearrange(text_query, 'L B H D -> B H L D') # [25, 2, 32, 128])

        vision_kv = rearrange(vision_features, 'BL Lv (H KV D) -> KV BL H Lv D', KV=2, H=self.num_key_value_heads)
        vision_key = vision_kv[0].contiguous() # [b h s d]
        vision_value = vision_kv[1].contiguous()
        
        vision_key = repeat_kv(vision_key, self.num_key_value_groups)
        vision_value = repeat_kv(vision_value, self.num_key_value_groups)
        
        # expend_cross_attn_mask
        attention_mask = text2vision_cross_attn_mask[:, None, :].repeat(1, text_state.shape[0], 1) 
        vision_context = self.cross_attn_core_attention(text_query, vision_key, vision_value, attn_mask=attention_mask).transpose(0, 1)

        # mask out the output if a sample is pure text
        vision_context = all_text_mask[None, :, None] * vision_context

        # Apply dynamic gate
        text_state = text_state + vision_context * gate_value
        
        return text_state

    def onlytext2media_cross_attn(self, 
                              text_state, 
                              text_query,
                              vision_features,
                              token_type,
                              text2vision_cross_attn_mask=None,
                              all_text_mask=None):
        '''
        text_query: [bs n h d]
        text_state: [bs n d]
        vision_features: [bs, vis_n, d]
        token_type: [bs, n]
        '''        
        # if vision_features is None or (self.is_hyper_enabled == False) or (all_text_mask.sum() == 0):
        if vision_features is None or (self.is_hyper_enabled == False):
            return text_state
        
        # select all the pure text token
        pure_text_query = []
        text_mask = ((token_type - 2) <= 0).bool()
        
        if "masksystem" in self.cross_attention_implementation:
            new_text_masks = [] 
            for idx, text_query_ in enumerate(text_query):
                # mask out all the tokens before the media
                first_im_token = (token_type[idx] == 3).nonzero()
                if len(first_im_token) == 0:
                    start = 0
                else:
                    start = first_im_token[0]
                text_mask_ = text_mask[idx].clone()
                text_mask_[:start] = False
                pure_text_query.append(text_query_[text_mask_])
                new_text_masks.append(text_mask_) 
            text_mask = torch.stack(new_text_masks, dim=0)
        else:
            for idx, text_query_ in enumerate(text_query):
                pure_text_query.append(text_query_[text_mask[idx]])

        # 2. pad all the text tokens
        text_query = torch.nn.utils.rnn.pad_sequence(pure_text_query, batch_first=True)
        padding_attn_mask = torch.ones(text_query.shape[:-2], dtype=torch.bool, device=text_state.device)
        for i, tensor in enumerate(pure_text_query):
            padding_attn_mask[i, len(tensor):] = False  # Mark padded elements as False

        B_c, L_c = text_query.shape[:2]
        D_head = self.head_dim
        
        # obtain dynamic gate value
        gate_value = self.cross_attn_gate_proj(text_state[text_mask]) # n, D
        if "warmup" in self.gating_type:
            gate_value = gate_value * self.cross_attn_warm_up_gate.tanh()

        vision_features = vision_features.contiguous()
        vision_features_k = self.k_proj(vision_features)
        vision_features_v = self.v_proj(vision_features)
        vision_features = torch.cat([vision_features_k, vision_features_v], dim=-1)

        text_query = text_query.transpose(1, 2)

        vision_kv = rearrange(vision_features, 'BL Lv (H KV D) -> KV BL H Lv D', KV=2, H=self.num_key_value_heads)
        vision_key = vision_kv[0].contiguous() # [b h s d]
        vision_value = vision_kv[1].contiguous()
        
        vision_key = repeat_kv(vision_key, self.num_key_value_groups)
        vision_value = repeat_kv(vision_value, self.num_key_value_groups)
        
        # expend_cross_attn_mask
        attention_mask = text2vision_cross_attn_mask[:, None, :].repeat(1, text_query.shape[2], 1) .to(torch.bool)
        vision_context = self.cross_attn_core_attention(text_query, vision_key, vision_value, attn_mask=attention_mask)

        # mask out the output if a sample is pure text
        vision_context = all_text_mask[:, None, None] * vision_context
        
        # Apply dynamic gate
        extended_attn_output = torch.zeros_like(text_state, dtype=text_state.dtype, device=text_state.device)
        extended_attn_output[text_mask] = extended_attn_output[text_mask] + vision_context[padding_attn_mask] * gate_value
        text_state = text_state + extended_attn_output
        # NOTE Min: just equvalent to the following line. Avoid error under deepspeed zero3
        # text_state[text_mask] = text_state[text_mask] + vision_context[padding_attn_mask] * gate_value
            
        return text_state  
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        token_type: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text2visual_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        activation_mask=None # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            # logger.warning_once(
            #     f"The input hidden states seems to be silently casted in float32, this might be related to"
            #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            #     f" {target_dtype}."
            # )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
        
        attn_output = _flash_attention_forward(
            query_states, # bs, n, head, head_dim
            key_states,   
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        # text-to-image cross-attention
        ####
        all_text_mask = (token_type == 3).sum(dim=-1).bool() # [bs, ] if False, indicate that this sample contains no image input
        
        if activation_mask is not None:
            # 统一执行路径：始终执行 cross-attn，再用连续掩码进行插值融合，确保各 rank 参与相同计算图
            if self.cross_attention_implementation.startswith("vanilla"):
                attn_output_all = self.all2media_cross_attn(
                    attn_output.permute(1, 0, 2),
                    query_states.permute(1, 0, 2, 3),
                    visual_hidden_states,
                    text2visual_attention_mask,
                    all_text_mask,
                ).permute(1, 0, 2)
            elif self.cross_attention_implementation.startswith("text-only-vanilla"):
                attn_output_all = self.onlytext2media_cross_attn(
                    attn_output,
                    query_states,
                    visual_hidden_states,
                    token_type=token_type,
                    text2vision_cross_attn_mask=text2visual_attention_mask,
                    all_text_mask=all_text_mask,
                )
            else:
                raise NotImplementedError(f"cross-attention type {self.cross_attention_implementation} not implemented")

            # 将 batch 级激活掩码广播到 [B, 1, 1] 后进行线性插值
            mask = activation_mask
            if mask.dim() == 1:
                mask = mask.reshape(-1, 1, 1)
            attn_output = attn_output * (1.0 - mask) + attn_output_all * mask

        else:
            if self.cross_attention_implementation.startswith("vanilla"): # all tokens can attend to the slow tokens
                attn_output = self.all2media_cross_attn(attn_output.permute(1, 0, 2), 
                                                        query_states.permute(1, 0, 2, 3),
                                                        visual_hidden_states, 
                                                        text2visual_attention_mask,
                                                        all_text_mask)
                attn_output = attn_output.permute(1,0,2)
            
            elif self.cross_attention_implementation.startswith("text-only-vanilla"): # only text tokens are allowed to attend the slow tokens
                attn_output = self.onlytext2media_cross_attn(attn_output, 
                                                         query_states, 
                                                         visual_hidden_states,
                                                         token_type=token_type,
                                                         text2vision_cross_attn_mask=text2visual_attention_mask,
                                                         all_text_mask=all_text_mask
                                                         )
            else:
                raise NotImplementedError(f"cross-attention type {self.cross_attention_implementation} not implemented")      
   
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class Qwen3SdpaAttention(Qwen3Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """        
    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            # logger.warning_once(
            #     "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            #     'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.view(*input_shape, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

# TODO: Min: Not implementated yet
class Qwen3HybridSdpaAttention(Qwen3SdpaAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, 
                 is_hyper_enabled, 
                 gating_type, 
                 cross_attn_implementation, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.is_hyper_enabled = is_hyper_enabled

        if self.is_hyper_enabled:
            self.gating_type = gating_type
            self.cross_attention_implementation = cross_attn_implementation
            self.cross_attn_kv_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim * 2, bias=True)
            
            if gating_type.startswith("whole-dynamic"):
                if "tanh" in gating_type:
                    self.cross_attn_gate_proj = nn.Sequential(
                        nn.Linear(self.hidden_size, 1),
                        nn.Tanh()
                    )
                else:
                    self.cross_attn_gate_proj = nn.Sequential(
                        nn.Linear(self.hidden_size, 1),
                    )
                    
                if gating_type.endswith("warmup"):
                    self.cross_attn_warm_up_gate = torch.nn.Parameter(torch.zeros(1))
                
            if "flashattn" in self.cross_attention_implementation:
                self.cross_attn_core_attention = FlashAttnCrossAttention(layer_number=-1, attention_dropout=self.attention_dropout)
            else:
                self.cross_attn_core_attention = ScaleDotProductCrossAttention(layer_number=-1, attention_dropout=self.attention_dropout)

    def text2media_cross_attn(self, 
                              text_state, 
                              text_query,
                              vision_features,
                              text2vision_cross_attn_mask=None,
                              all_text_mask=None):
        '''
        text_query: [s b h d]
        text_state: s b d
        vision_features: [num_vis, b,  d]
        '''        
        if vision_features is None or (self.is_hyper_enabled == False):
            return text_state
        
        # obtain dynamic gate value
        L_c, B_c = text_state.shape[:2]
        D_head = self.head_dim
        
        gate_value = rearrange(
            self.gate_proj(
                rearrange(text_state, 'L B (Head D) -> (L B Head) D', D=D_head)),
            '(L B Head) D -> L B (Head D)', L=L_c, B=B_c)

        vision_features = vision_features.contiguous()
        vision_features = self.v_kv_proj(vision_features)
        # length_each_img = vision_features.shape[1]
        # sequence_length = text_query.shape[0]            
        query_layer = rearrange(query_layer, 'L B H D -> B H L D') # [25, 2, 32, 128])

        vision_kv = rearrange(vision_features, 'BL Lv (H KV D) -> KV 1 H (BL Lv) D', KV=2, H=self.num_key_value_heads)
        vision_key = vision_kv[0].contiguous() # [b h s d]
        vision_value = vision_kv[1].contiguous()
        
        # Apply MI-Rope
        # key_layer = self.apply_mi_rope(key_layer, media_offset_line=self.visual_cache['media_offset'][batch_id,:,1]-curr_offset[0], length_each_img=length_each_img)
        key_layer = repeat_kv(key_layer, self.num_key_value_groups)
        value_layer = repeat_kv(value_layer, self.num_key_value_groups)
        vision_context = self.v_core_attention_sdpa(query_layer, vision_key, vision_value, attn_mask=None, order='bhsd').squeeze(1) # TODO

        # Apply dynamic gate
        text_state = text_state * (1 - gate_value) + vision_context * gate_value
        
        return text_state
    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        token_type: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text2visual_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            # logger.warning_once(
            #     "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            #     'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.view(*input_shape, -1)
        
        # text-to-image cross-attention
        ####
        all_text_mask = (token_type == 3).sum(dim=-1).bool() # [bs, ] if False, indicate that this sample contains no image input
        if self.cross_attention_implementation.startswith("vanilla"):
            attn_output = self.text2media_cross_attn(attn_output.permute(1, 0, 2), 
                                                     query_states.permute(1, 0, 2, 3),
                                                     visual_hidden_states, 
                                                     text2visual_attention_mask,
                                                     all_text_mask)
            attn_output = attn_output.permute(1,0,2)
            
        elif self.cross_attention_implementation.startswith("text-only-vanilla"):
            attn_output = self.onlytext2media_cross_attn(attn_output, 
                                                         query_states, 
                                                         visual_hidden_states,
                                                         token_type=token_type,
                                                         text2vision_cross_attn_mask=text2visual_attention_mask,
                                                         all_text_mask=all_text_mask
                                                         )
        else:
            raise NotImplementedError(f"cross-attention type {self.cross_attention_implementation} not implemented")
        ####          

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN3_ATTENTION_CLASSES = {
    "eager": Qwen3Attention,
    "flash_attention_2": Qwen3FlashAttention2,
    "sdpa": Qwen3SdpaAttention,
}

QWEN3_HYBRID_ATTENTION_CLASSES = {
    "flash_attention_2": Qwen3HybridFlashAttention2,
    "sdpa": Qwen3HybridSdpaAttention, # Not implemented yet, only support flash attn 
}


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            # logger.warning_once(
            #     f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
            #     "unexpected results may be encountered."
            # )
            pass
        self.self_attn = QWEN3_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3HybridDecoderLayer(nn.Module):
    def __init__(self, 
                 config, 
                 layer_idx: int, 
                 
                 is_hyper_enabled=False, 
                 cross_attn_implementation="vanilla", # in ['vanilla' and 'text-only-vanilla']
                 cross_attn_gating_type="channel-wise-dynamic-sigmoid"):
        super().__init__()
        self.is_hyper_enabled = is_hyper_enabled
        
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            # logger.warning_once(
            #     f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
            #     "unexpected results may be encountered."
            # )
            pass
        
        self.self_attn = QWEN3_HYBRID_ATTENTION_CLASSES[config._attn_implementation](config=config,
                                                                                    layer_idx=layer_idx, 
                                                                                    is_hyper_enabled=is_hyper_enabled, 
                                                                                    cross_attn_implementation=cross_attn_implementation,
                                                                                    gating_type=cross_attn_gating_type)
        

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False # move the gradient checkpointing to the forward function of attn and MLP
        self.activate = False

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, 
                        vis_x,
                        cross_attn_mask=None, 
                        token_type=None):
        
        self.vis_x = vis_x
        self.cross_attn_mask = cross_attn_mask
        self.media_locations = token_type
    
    def clear_vis_x(self):
        self.vis_x = None
        self.cross_attn_mask = None
        self.media_locations = None

    def mlp_forward(self, hidden_states):
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        activation_mask=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # process image embedding
        visual_tokens = self.vis_x
        cross_attn_mask = self.cross_attn_mask
        token_type = self.media_locations
        visual_tokens = self.input_layernorm(visual_tokens)

        # Self Attention
        if self.gradient_checkpointing and self.training:
            hidden_states, self_attn_weights, present_key_value = torch.utils.checkpoint.checkpoint(
                self.self_attn,
                hidden_states,
                visual_tokens,
                token_type,
                attention_mask,
                cross_attn_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                activation_mask
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                visual_hidden_states=visual_tokens,
                text2visual_attention_mask=cross_attn_mask,
                token_type=token_type,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                activation_mask=activation_mask
            )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.mlp_forward,
                hidden_states)
        else:
            hidden_states = self.mlp_forward(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Predictor(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=2560, routing_ratio=0.25):
        super().__init__()
        self.routing_ratio = routing_ratio
        self.visual_in_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//4),
            nn.GELU()
        )

        self.text_in_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//4),
            nn.GELU()
        )
        

        self.out_proj = nn.Sequential(
            nn.Linear((embed_dim // 4 * 3), embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.GELU(),
            nn.Linear(embed_dim // 8, 2),
            nn.LogSoftmax(dim=-1)
        )
        
        # 初始化 Predictor 网络权重
        for module in [self.visual_in_proj, self.text_in_proj, self.out_proj]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, visual_tokens, text_tokens):
        visual_tokens = self.visual_in_proj(visual_tokens)
        text_tokens = self.text_in_proj(text_tokens)
        B, N, C = visual_tokens.size()
        visual_tokens_frame_level = visual_tokens.mean(dim=1, keepdim=True)
        x = torch.cat([visual_tokens, visual_tokens_frame_level.expand(B, N, C)], dim=-1)
        x = x.flatten(0,1)
        x = torch.cat([x, text_tokens.mean(dim=0,  keepdim=True).expand(B*N, C)], dim=-1)
        scores = self.out_proj(x)
        
        # 添加数值稳定性检查
        if self.training:
            if torch.isnan(scores).any():
                print(f"[Predictor] Warning: NaN detected in scores")
                scores = torch.zeros_like(scores)
            if torch.isinf(scores).any():
                print(f"[Predictor] Warning: Inf detected in scores")
                scores = torch.clamp(scores, -10, 10)

        if self.training:
            # 训练时：使用温度缩放的 softmax + Top-K
            if hasattr(self, 'training_step'):
                tau = max(0.1, 1.0 * (0.9 ** self.training_step))
            else:
                tau = 1.0
            
            # 温度缩放的 softmax，添加数值稳定性
            scores_scaled = scores / tau
            # 减去最大值避免数值溢出
            scores_scaled = scores_scaled - scores_scaled.max(dim=-1, keepdim=True)[0]
            probs = F.softmax(scores_scaled, dim=-1)
            
            # 根据 routing_ratio 选择 token 数量
            num_text_related_tokens = int(B*N*self.routing_ratio)
            topk_probs, topk_indices = torch.topk(probs[:, 0], num_text_related_tokens, dim=-1)
            
            # 创建软掩码（用于梯度传播）
            soft_mask = torch.zeros_like(probs[:, 0])
            soft_mask.scatter_(-1, topk_indices, topk_probs)
            
            # 创建硬掩码（用于前向传播）
            hard_mask = torch.zeros_like(probs[:, 0])
            hard_mask.scatter_(-1, topk_indices, 1.0)
            
            # Straight-Through 技巧
            text_related_tokens_mask = hard_mask.detach() + soft_mask - soft_mask.detach()
        else:
            probs = F.softmax(scores, dim=-1)[:, 0]
            num_text_related_tokens = int(B*N*self.routing_ratio)
            topk_probs, topk_indices = torch.topk(probs, num_text_related_tokens, dim=-1)
            text_related_tokens_mask = torch.zeros_like(probs, dtype=torch.bool)
            text_related_tokens_mask.scatter_(-1, topk_indices, 1)  # 将选中的位置设为1

      # 训练时返回float以保留梯度；推理时返回bool掩码
        return text_related_tokens_mask if self.training else text_related_tokens_mask.to(torch.bool)


class DynamicRouter(nn.Module):
    def __init__(self, d_model, num_layers, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        # 用输入特征的均值作为路由输入
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),  # 添加 LayerNorm 稳定训练
            nn.ReLU(),
            nn.Linear(d_model//2, d_model//4),
            nn.LayerNorm(d_model//4),  # 添加 LayerNorm 稳定训练
            nn.ReLU(),
            nn.Linear(d_model//4, num_layers)  # 输出每层的得分
        )
        
        # 初始化路由网络权重，使用较小的标准差避免梯度爆炸
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                # 使用 Xavier 初始化，但标准差更小
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        
        # 负载均衡辅助损失
        self.aux_loss = 0
        

    def forward(self, x):
        """ x: 输入特征 [batch, seq_len, d_model] """
        # 计算路由得分 [batch, num_layers]
        pooled = x.mean(dim=1)  # [batch, d_model]
        # 确保数据类型匹配
        pooled = pooled.to(next(self.router.parameters()).dtype)
        scores = self.router(pooled)  # [batch, num_layers]
        
        # 添加数值稳定性检查和梯度监控
        if self.training:
            if torch.isnan(scores).any():
                print(f"[DynamicRouter] Warning: NaN detected in scores")
                scores = torch.zeros_like(scores)
            if torch.isinf(scores).any():
                print(f"[DynamicRouter] Warning: Inf detected in scores")
                scores = torch.clamp(scores, -10, 10)
            
            # 监控得分范围，如果过大则进行裁剪
            # if scores.abs().max() > 2.0:
            #     print(f"[DynamicRouter] Warning: Large scores detected: {scores.abs().max()}")
            #     scores = torch.clamp(scores, -2.0, 2.0)
        
        # Top-K 路由选择
        if self.training:
            # 训练时：使用温度缩放的 softmax + Top-K
            if hasattr(self, 'training_step'):
                tau = max(0.1, 1.0 * (0.9 ** self.training_step))
            else:
                tau = 1.0
            
            # 温度缩放的 softmax，添加数值稳定性
            scores_scaled = scores / tau
            # 减去最大值避免数值溢出
            scores_scaled = scores_scaled - scores_scaled.max(dim=-1, keepdim=True)[0]
            probs = F.softmax(scores_scaled, dim=-1)
            
            # Top-K 选择
            topk_probs, topk_indices = torch.topk(probs, self.num_experts, dim=-1)
            
            # 创建软掩码（用于梯度传播）
            soft_mask = torch.zeros_like(probs)
            soft_mask.scatter_(-1, topk_indices, topk_probs)
            
            # 创建硬掩码（用于前向传播）
            hard_mask = torch.zeros_like(probs)
            hard_mask.scatter_(-1, topk_indices, 1.0)
            
            # Straight-Through 技巧
            activation_mask = hard_mask.detach() + soft_mask - soft_mask.detach()
        else:
            probs = F.softmax(scores, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, self.num_experts, dim=-1)
            activation_mask = torch.zeros_like(probs)
            activation_mask.scatter_(-1, topk_indices, 1)  # 将选中的位置设为1
        
        # 计算负载均衡损失（防止专家垄断）
        #self.aux_loss = self._load_balancing_loss(probs, activation_mask)
        
        return activation_mask  # [batch, num_layers] 其中num_experts个位置为1

    def _load_balancing_loss(self, probs, mask):
        # 参考Switch Transformer的负载均衡损失
        P = probs.sum(0) / probs.size(0)  # 平均激活概率
        U = mask.sum(0) / mask.size(0)    # 实际激活频率
        return (P * U).sum() * self.num_layers

