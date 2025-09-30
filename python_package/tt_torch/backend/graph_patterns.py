# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch.fx import Graph, GraphModule, Node
from torch.fx import symbolic_trace
from torch.fx.subgraph_rewriter import ReplacedPatterns, _replace_attributes
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
import torch.fx as fx
import copy
import torch.nn as nn


def lower_modules_to_functionals(gm: fx.GraphModule) -> fx.GraphModule:
    g = gm.graph
    for n in list(g.nodes):
        if n.op != "call_module":
            continue
        mod = gm.get_submodule(n.target)
        if isinstance(mod, nn.Linear):
            x = n.args[0]
            with g.inserting_before(n):
                w = g.get_attr(f"{n.target}.weight")
                b = g.get_attr(f"{n.target}.bias") if mod.bias is not None else None
                new = g.call_function(F.linear, (x, w, b))

            n.replace_all_uses_with(new)
            g.erase_node(n)
            continue
        if isinstance(mod, nn.SiLU):
            x0 = n.args[0]
            # x1 = n.kwargs.get('inplace', False)
            kwargs = dict(n.kwargs)
            if "inplace" not in kwargs:
                kwargs["inplace"] = False
            with g.inserting_before(n):
                new = g.call_function(F.silu, (x0,), kwargs=kwargs)
            n.replace_all_uses_with(new)
            g.erase_node(n)
    gm.recompile()


class AssertFreeSubgraphMatcher(SubgraphMatcher):
    def __init__(
        self,
        pattern: Graph,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
    ) -> None:
        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.remove_overlapping_matches = remove_overlapping_matches
        self.ignore_literals = ignore_literals

        self.pattern_placeholder_nodes = [
            n for n in pattern.nodes if n.op == "placeholder"
        ]
        output_node = next(iter(reversed(pattern.nodes)))
        # nodes returned by outputs
        self.pattern_returning_nodes: list[Node] = output_node.all_input_nodes

        self.pattern_anchors: list[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # If a node has output_node as the ONLY user, then this node is a graph sink,
            # and should be matched against as an anchor
            self.pattern_anchors = [
                n for n in output_node.all_input_nodes if len(n.users) == 1
            ]

    @override
    def _is_contained(self, nodes_map):
        return True


def nuke_nodes_with_all_users(gm, roots):
    visited = set()
    postorder = []

    def dfs(n):
        if n in visited:
            return
        visited.add(n)
        for u in list(n.users):
            dfs(u)
        postorder.append(n)

    for r in roots:
        dfs(r)
    for n in postorder:
        try:
            gm.graph.erase_node(n)
        except Exception:
            pass
    try:
        gm.recompile()
    except Exception:
        pass


def replace_pattern(gm, pattern, replacement):
    lower_modules_to_functionals(gm)

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph

    matcher = AssertFreeSubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=True,
    )

    matches = matcher.match(gm.graph)

    match_changed_node = {}
    replaced = []

    if isinstance(replacement, GraphModule):
        common_replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        common_replacement_graph = replacement
    elif callable(replacement):
        common_replacement_graph = symbolic_trace(replacement).graph
    else:
        assert False, "replacement must be a GraphModule, Graph, or callable"

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: dict[Node, Node] = {}

    match_and_replacements = []
    for match in matches:
        # map placeholders
        repl_ph = [n for n in common_replacement_graph.nodes if n.op == "placeholder"]
        assert len(repl_ph) == len(match.placeholder_nodes)
        val_map = {}
        for rn, gn in zip(repl_ph, match.placeholder_nodes):
            val_map[rn] = match_changed_node.get(gn, gn)

        user_nodes = set()
        for n in match.returning_nodes:
            user_nodes.update(n.users)

        first_user = None
        if len(user_nodes) == 1:
            first_user = next(iter(user_nodes))
        elif len(user_nodes) > 1:
            for n in gm.graph.nodes:
                if n in user_nodes:
                    first_user = n
                    break

        first_next = None
        if first_user is None:
            next_node = None
            for n in reversed(list(gm.graph.nodes)):
                if n in match.returning_nodes:
                    first_next = next_node
                    break
                next_node = n
        insert_point = first_user if first_user is not None else first_next
        assert insert_point is not None

        with gm.graph.inserting_before(insert_point):
            copied_returns = gm.graph.graph_copy(common_replacement_graph, val_map)

        if isinstance(copied_returns, Node):
            copied_returns = (copied_returns,)

        # replace uses
        assert len(match.returning_nodes) == len(copied_returns)
        for old, new in zip(match.returning_nodes, copied_returns):
            old.replace_all_uses_with(new)
            match_changed_node[old] = new

        roots = []
        for pn in pattern_graph.nodes:
            if pn.op not in ("placeholder", "output"):
                gn = match.nodes_map.get(pn)
                if gn is not None:
                    roots.append(gn)

        nuke_nodes_with_all_users(gm, roots)

        replaced.append(match)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replaced,
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return match_and_replacements


def get_patterns():
    rewrite_patterns = {
        "sparse_moe_block": (
            sparse_moe_block_pattern,
            replacement_sparse_moe_block_pattern,
        )
    }
    return rewrite_patterns


def sparse_moe_block_pattern(
    hidden_states,  # (N, H) or (B*T, H) flattened tensor
    hidden_dim,  # H
    routing_weights,  # (N, K), indexed to (n_e,1)
    expert_mask_slice,  # expert_mask[expert_idx] (corresponds to getitem_2 in original graph)
    w1,
    w2,
    w3,  # weights inside expert_layer (w1,w3 -> gate/up, w2 -> down)
    final_hidden_states,  # (N, H) final aggregated result (if None, initialized to 0 inside)
):

    # where
    where = torch.where(expert_mask_slice)  # call_function where
    idx = where[0]  # getitem_42
    top_x = where[1]  # getitem

    # hidden_states[None, top_x].reshape(-1, H)
    cur = hidden_states[None, top_x].reshape(-1, hidden_dim)  # getitem_5

    # linear_1 -> silu -> linear_2 -> mul
    a = F.linear(cur, w1, None)  # linear_1
    a = F.silu(
        a,
    )  # silu (inplace=False)
    b = F.linear(cur, w3, None)  # linear_2
    prod = a * b  # current_hidden_states

    # down-proj
    out_e = F.linear(prod, w2, None)  # current_hidden_states_1

    contrib = out_e * routing_weights[top_x, idx, None]  # current_hidden_states_2
    contrib = contrib.to(hidden_states.dtype)  # call_method to

    final_hidden_states = final_hidden_states.index_add_(
        0, top_x, contrib
    )  # call_method index_add_
    return final_hidden_states


def replacement_sparse_moe_block_pattern(
    hidden_states,
    hidden_dim,
    routing_weights,
    expert_mask_slice,
    w1,
    w2,
    w3,
    final_hidden_states,
):
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
    a = F.linear(hidden_states, w1, bias=None)  # (N, F)
    b = F.linear(hidden_states, w3, bias=None)  # (N, F)
    tmp = F.silu(a) * b  # (N, F)
    out_e = F.linear(tmp, w2, bias=None)  # (N, H)

    # Per-token weight for this expert (only selected tokens >0)
    mask_T = expert_mask_slice.permute(1, 0).to(routing_weights.dtype)  # (N, K)
    w_e = (routing_weights * mask_T).sum(dim=1)  # (N,)

    # Weighted contribution calculation
    contrib = out_e * w_e.unsqueeze(1).to(out_e.dtype)  # (N, H)

    # Accumulate and return (dtype/device aligned)
    final_hidden_states.add_(contrib.to(final_hidden_states.dtype))
    return final_hidden_states
