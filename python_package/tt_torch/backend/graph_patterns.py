import torch
import torch.nn.functional as F

def get_patterns():
    rewrite_patterns = {
        "sparse_moe_block": (sparse_moe_block_pattern, replacement_sparse_moe_block_pattern)
    }
    return rewrite_patterns

def sparse_moe_block_pattern(
    hidden_states,         # (N, H) or (B*T, H) flattened tensor
    hidden_dim,            # H
    routing_weights,       # (N, K), indexed to (n_e,1)
    expert_mask_slice,     # expert_mask[expert_idx] (corresponds to getitem_2 in original graph)
    w1, w2, w3,            # weights inside expert_layer (w1,w3 -> gate/up, w2 -> down)
    final_hidden_states    # (N, H) final aggregated result (if None, initialized to 0 inside)
):

    # where
    where = torch.where(expert_mask_slice)        # call_function where
    idx    = where[0]                             # getitem_42
    top_x  = where[1]                             # getitem

    # hidden_states[None, top_x].reshape(-1, H)
    cur = hidden_states[None, top_x].reshape(-1, hidden_dim)          # getitem_5

    # linear_1 -> silu -> linear_2 -> mul
    a = F.linear(cur, w1, None)              # linear_1
    a = F.silu(a,)                  # silu (inplace=False)
    b = F.linear(cur, w3, None)              # linear_2
    prod = a * b                                  # current_hidden_states

    # down-proj
    out_e = F.linear(prod, w2, None)         # current_hidden_states_1

    contrib = out_e * routing_weights[top_x, idx, None]                             # current_hidden_states_2
    contrib = contrib.to(hidden_states.dtype)      # call_method to

    final_hidden_states = final_hidden_states.index_add_(0, top_x, contrib)             # call_method index_add_
    return final_hidden_states

def replacement_sparse_moe_block_pattern(hidden_states, hidden_dim, routing_weights, expert_mask_slice, w1, w2, w3, final_hidden_states):
    """
    hidden_states      : (N, H)
    w1, w3             : (F, H)  # F.linear weight convention
    w2                 : (H, F)
    routing_weights    : (N, K)
    expert_mask_slice  : (K, N)  # {0,1}
    final_hidden_states: (N, H)  # accumulation buffer

    return             : (N, H)  # updated accumulation buffer
    """
    # Expert MLP forward
    a   = F.linear(hidden_states, w1, bias=None)   # (N, F)
    b   = F.linear(hidden_states, w3, bias=None)   # (N, F)
    tmp = F.silu(a) * b                            # (N, F)
    out_e = F.linear(tmp, w2, bias=None)           # (N, H)

    # Per-token weight for this expert (only selected tokens >0)
    mask_T = expert_mask_slice.permute(1, 0).to(routing_weights.dtype)  # (N, K)
    w_e = (routing_weights * mask_T).sum(dim=1)                          # (N,)

    # Weighted contribution calculation
    contrib = out_e * w_e.unsqueeze(1).to(out_e.dtype)                   # (N, H)

    # Accumulate and return (dtype/device aligned)
    final_hidden_states.add_(contrib.to(final_hidden_states.dtype))
    return final_hidden_states
