# Copyright 2025 IsNoobGrammer. All Rights Reserved.
# Apache License 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd.xla_sharding as xs
from typing import Tuple, Optional
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

SDPA_ATTENTION_AVAILABLE = True

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value states for Grouped Query Attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class XLASDPAWrapper(nn.Module):
    def __init__(self, original_attention, mesh, partition_spec, rotary_func=apply_rotary_pos_emb):
        """Initialize the SDPA wrapper for XLA and TPU optimization using native math implementation.

        Args:
            original_attention: The original attention module to wrap.
            mesh: The TPU mesh for sharding.
            partition_spec: The partition specification for SPMD.
            rotary_func: Function to apply rotary positional embeddings (default: apply_rotary_pos_emb).
        """
        super().__init__()
        self.original_attention = original_attention
        self.mesh = mesh
        self.partition_spec = partition_spec
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # Compute groups for GQA
        self.head_dim = original_attention.head_dim
        self.hidden_size = original_attention.config.hidden_size
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx
        self.rotary_func = rotary_func
        
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional['Cache'] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with SDPA optimized for XLA and TPUs using native math implementation.

        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings: Tuple of (cos, sin) for rotary embeddings.
            attn_mask: Optional attention mask for SDPA (if provided, used instead of causal).
            past_key_value: Optional cache for key/value states.
            cache_position: Optional tensor specifying cache positions.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (attention output, None) for compatibility with original attention.
        """
        bsz, q_len, _ = hidden_states.size()
        

        
        query = self.original_attention.q_proj(hidden_states)
        key = self.original_attention.k_proj(hidden_states)
        value = self.original_attention.v_proj(hidden_states)
        
        


        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)


        # xs.mark_sharding(query,self.mesh,self.partition_spec)
        # xs.mark_sharding(key,self.mesh,self.partition_spec)
        # xs.mark_sharding(value,self.mesh,self.partition_spec)

        
        
        cos, sin = position_embeddings
        query, key = self.rotary_func(query, key, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_value.update(key, value, self.layer_idx, cache_kwargs)


        
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        is_gqa = self.num_kv_groups > 1 
        
        with sdpa_kernel([SDPBackend.MATH]):
            attn_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,  
                is_causal=True ,  
                scale=self.scaling,
                enable_gqa=is_gqa
            )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None
    




