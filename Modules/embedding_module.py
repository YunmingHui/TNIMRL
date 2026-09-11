import torch
from torch import nn
import numpy as np

from .utils import MergeLayer


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, memory_dimension, n_time_features, output_dim,
                 device, n_layers=1, n_heads=2, dropout=0.1):
        super(GraphAttentionEmbedding, self).__init__()
        assert (n_layers >= 0)
        # Layer i takes input from layer i-1:
        #   i=0 (innermost): input is raw memory → dim = memory_dim + time_dim
        #   i>0 (outer):     input is previous attention output → dim = output_dim + time_dim
        self.attention_models = torch.nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                qkv_dim = memory_dimension + n_time_features
            else:
                qkv_dim = output_dim + n_time_features
            self.attention_models.append(TemporalAttentionLayer(
                query_dim=qkv_dim,
                key_dim=qkv_dim,
                value_dim=qkv_dim,
                output_dim=output_dim,
                n_head=n_heads,
                dropout=dropout,
            ))
        self.device = device

    def compute_embedding(self, source_nodes, nodes_memory, neighbor_finder,
                          final_timestamp: int, time_encoder,
                          n_layers=1, n_neighbors=20,
                          node_time_memory_fn=None, query_times=None):
        """
        Args:
            node_time_memory_fn: callable(node_ids: list, times: list) -> Tensor [N, D]
                If provided (ablation mode), base case uses this to get s_j(t_k)
                instead of the global nodes_memory (final memory).
            query_times: np.array [N] of float, the timestamp at which each node
                in source_nodes is being queried. None means final_timestamp for all.
        """

        # ------------------------------------------------------------------ #
        # Base case: return memory at query time (ablation) or final memory   #
        # ------------------------------------------------------------------ #
        if n_layers == 0:
            if node_time_memory_fn is not None and query_times is not None:
                # Ablation: use s_j(t_k) — memory at interaction time
                return node_time_memory_fn(list(source_nodes), list(query_times))
            # Original / your method: use cached final memory
            return nodes_memory[source_nodes]

        # ------------------------------------------------------------------ #
        # Recursive case                                                       #
        # ------------------------------------------------------------------ #
        num_nodes = len(source_nodes)

        # query_times for source nodes: final_timestamp if not specified
        if query_times is None:
            query_times = np.ones(num_nodes) * final_timestamp

        # Recurse for source nodes (at their own query times)
        source_node_memory = self.compute_embedding(
            source_nodes, nodes_memory, neighbor_finder, final_timestamp, time_encoder,
            n_layers=n_layers - 1, n_neighbors=n_neighbors,
            node_time_memory_fn=node_time_memory_fn, query_times=query_times,
        )

        # Time encoding for source nodes at final_timestamp
        final_ts = torch.from_numpy(
            np.ones(num_nodes) * final_timestamp
        ).float().to(self.device)
        final_ts_encoding = time_encoder(final_ts.unsqueeze(dim=1)).view(num_nodes, -1)
        source_node_embedding = torch.cat((source_node_memory, final_ts_encoding), dim=1)
        query = torch.unsqueeze(source_node_embedding, dim=1).permute([1, 0, 2])

        # Get temporal neighbors (at final_timestamp, same as original)
        neighbors, _, edge_times = neighbor_finder.get_temporal_neighbor(
            list(range(num_nodes)),
            [final_timestamp] * num_nodes,
            n_neighbors=n_neighbors,
        )

        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        mask = neighbors_torch == 0

        edge_times_torch = torch.from_numpy(edge_times).float().to(self.device)
        edge_time_encoding = time_encoder(edge_times_torch)

        # KEY CHANGE for ablation:
        # Neighbors are queried at their interaction time t_k (edge_times),
        # not at final_timestamp. This is passed as query_times in the recursive call.
        neighbor_query_times = edge_times.flatten()  # [num_nodes * n_neighbors]

        neighbor_memory = self.compute_embedding(
            neighbors.flatten(), nodes_memory, neighbor_finder, final_timestamp, time_encoder,
            n_layers=n_layers - 1, n_neighbors=n_neighbors,
            node_time_memory_fn=node_time_memory_fn, query_times=neighbor_query_times,
        )
        neighbor_memory = neighbor_memory.view(num_nodes, n_neighbors, -1)
        key = torch.cat((edge_time_encoding, neighbor_memory), dim=2).permute([1, 0, 2])

        source_memory = self.aggregate(n_layers, query, key, key, mask)
        return source_memory

    def aggregate(self, n_layer, query, key, value, mask):
        attention_model = self.attention_models[n_layer - 1]
        source_embedding, _ = attention_model(query, key, value, mask)
        return source_embedding


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, n_head=2, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.multi_head_target = nn.MultiheadAttention(embed_dim=query_dim,
                                                       kdim=key_dim,
                                                       vdim=value_dim,
                                                       num_heads=n_head,
                                                       dropout=dropout)
        self.merger = MergeLayer(query_dim, query_dim, query_dim, output_dim)

    def forward(self, query, key, value, neighbors_padding_mask):
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

        attn_output, attn_output_weights = self.multi_head_target(
            query=query, key=key, value=value,
            key_padding_mask=neighbors_padding_mask,
        )

        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        attn_output = self.merger(attn_output, query.squeeze(0))

        return attn_output, attn_output_weights