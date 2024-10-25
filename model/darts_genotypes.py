from collections import namedtuple
import torch
import torch.nn as nn
from model import darts_operation as ops

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none']

PRIMITIVES_ISTA = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5']

def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(op, ops.DropPath_())
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)
    return dag

def from_str(s):
    genotype = eval(s)
    return genotype

def parse(alpha, k):
    # [2, 5], [3, 5], [4, 5], [5, 5]
    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'
    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha: 
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        # edge_max: Tensor(n_edges, 1)
        # primitive_indices: Tensor(n_edges, 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        # only can take two edges among 
        # topk_edge_values: Tensor(2)
        # topk_edge_indices: Tensor(2)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))
        gene.append(node_gene)
        # (nodes+2, n_edges, best_ops)
    return gene