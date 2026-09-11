import torch
import numpy as np
from collections import defaultdict
from torch_geometric.nn import Set2Set

from Modules.message_aggregator import message_aggregator
from Modules.memory_updater import MemoryUpdater
from Modules.memory import Memory
from Modules.embedding_module import GraphAttentionEmbedding
from Modules.time_encoding import TimeEncode


class RLDIM(torch.nn.Module):
    def __init__(
        self, memory_dimension, time_encoder_dimension, device, number_nodes, number_embedding_layer=1
    ):
        super(RLDIM, self).__init__()

        self.device = device
        self.number_embedding_layer = number_embedding_layer
        self.time_encoder_dimension = time_encoder_dimension
        self.memory_dimension = memory_dimension
        self.message_dimension = 2 * memory_dimension + time_encoder_dimension

        self.memory_updater = MemoryUpdater(
            message_dimension=self.message_dimension,
            memory_dimension=self.memory_dimension,
            device=self.device,
        )
        self.time_encoder = TimeEncode(self.time_encoder_dimension).to(self.device)

        embed_dim = memory_dimension + time_encoder_dimension

        self.attention_embedding = GraphAttentionEmbedding(
            memory_dimension=memory_dimension,
            n_time_features=time_encoder_dimension,
            output_dim=embed_dim,
            device=device,
            n_layers=self.number_embedding_layer
        ).to(self.device)

        self.set2set = Set2Set(in_channels=embed_dim, processing_steps=6).to(self.device)

        self.value_calculator = torch.nn.Sequential(
            torch.nn.Linear(4 * embed_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1)
        ).to(self.device)

        self.state_norm = torch.nn.LayerNorm(2 * embed_dim).to(self.device)
        self.seed_norm = torch.nn.LayerNorm(embed_dim).to(self.device)
        self.action_norm = torch.nn.LayerNorm(embed_dim).to(self.device)

        self.memory = Memory(
            n_nodes=number_nodes,
            memory_dimension=self.memory_dimension,
            device=self.device,
        )
        self.cached_memory = None

    def computer_memory(self, edge_list, batch_size, number_nodes):
        if self.cached_memory is not None:
            return self.cached_memory

        self.memory.__init_memory__()

        edge_list = [
            (edge[0], edge[1], edge[2], index) for index, edge in enumerate(edge_list)
        ]
        batches = [
            edge_list[i: i + batch_size] for i in range(0, len(edge_list), batch_size)
        ]
        for batch in batches:
            batch_source_nodes = []
            batch_destination_nodes = []
            batch_timestamps = []
            batch_edge_indexs = []
            for edge in batch:
                batch_source_nodes.append(edge[0])
                batch_destination_nodes.append(edge[1])
                batch_timestamps.append(edge[2])
                batch_edge_indexs.append(edge[3])
            unique_nodes, node_id_to_messages = self.get_raw_messages(
                batch_source_nodes + batch_destination_nodes,
                batch_destination_nodes + batch_source_nodes,
                np.array(batch_timestamps + batch_timestamps),
            )
            unique_nodes, unique_messages, unique_timestamps = message_aggregator(
                unique_nodes, node_id_to_messages
            )
            self.memory_updater.update_memory(
                self.memory, unique_nodes, unique_messages, timestamps=unique_timestamps
            )

        self.cached_memory = self.memory.get_memory(list(range(number_nodes))).detach()
        return self.cached_memory

    def compute_embedding(self, nodes_memory, neighbor_finder, final_timestamp):
        source_nodes = list(range(nodes_memory.size()[0]))
        nodes_embedding = self.attention_embedding.compute_embedding(
            source_nodes,
            nodes_memory,
            neighbor_finder,
            final_timestamp,
            self.time_encoder,
            n_layers=self.number_embedding_layer,
            n_neighbors=5,
        )
        return nodes_embedding

    def compute_graph_representation(self, embedding):
        graph_representation = self.set2set(embedding)
        return self.state_norm(graph_representation.view(-1))

    def calculate_value(self, embedding, graph_representation, state: list, actions: list):
        batch_size = len(actions)

        action_embeddings = self.action_norm(embedding[actions])  # [B, D]

        if state:
            seed_repr = self.seed_norm(embedding[state]).mean(dim=0)          # [D]
            attn_out = seed_repr.unsqueeze(0).expand(batch_size, -1)          # [B, D]
        else:
            attn_out = torch.zeros(batch_size, embedding.shape[1], device=embedding.device)

        global_expanded = graph_representation.unsqueeze(0).expand(batch_size, -1)
        combined_input = torch.cat([global_expanded, attn_out, action_embeddings], dim=1)
        q_values = self.value_calculator(combined_input)
        return q_values.squeeze()

    def calculate_value_batch(self, embedding, graph_representation, states: list, actions: list):
        """
        Batched Q-value for B (state_i, action_i) pairs.
        states:  list of B lists (variable-length seed sets)
        actions: list of B node indices
        Returns: [B] tensor
        """
        B = len(actions)
        D = embedding.shape[1]

        action_embeddings = self.action_norm(embedding[actions])  # [B, D]

        seed_reprs = torch.zeros(B, D, device=embedding.device)
        for i, s in enumerate(states):
            if s:
                seed_reprs[i] = self.seed_norm(embedding[s]).mean(dim=0)     # [D]

        global_expanded = graph_representation.unsqueeze(0).expand(B, -1)
        combined = torch.cat([global_expanded, seed_reprs, action_embeddings], dim=1)
        return self.value_calculator(combined).squeeze(-1)                    # [B]

    def get_raw_messages(self, source_nodes, destination_nodes, edge_times):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)

        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)
        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(
            source_time_delta.unsqueeze(dim=1)
        ).view(len(source_nodes), -1)

        source_message = torch.cat(
            [source_memory, destination_memory, source_time_delta_encoding], dim=1
        )
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages