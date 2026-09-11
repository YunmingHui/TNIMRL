from DiffusionModel.diffusion_model import SocialSIS
from Modules.utils import compute_time_statistics, Data, get_neighbor_finder
from tqdm import tqdm

import os
import numpy as np
import copy


class IMEnv:
    def __init__(
        self,
        dataset,
        edge_thresholds,
        activation_length_percent,
        activation_length,
        seed_set_szie,
        min_actived_nodes,
        reward_scale,
    ):
        edge_file_path = f"Data/{dataset}/{dataset}_edges_all.txt"
        if not os.path.exists(edge_file_path):
            raise Exception(f"Edge file {edge_file_path} dose not exist!")
        self.socialsis = SocialSIS(
            edge_file_path=edge_file_path,
            edge_thresholds=edge_thresholds,
            activation_length_percent=activation_length_percent,
            activation_length=activation_length,
        )

        self.num_nodes = self.socialsis.get_number_nodes()
        self.edges = self.socialsis.get_edges()
        self.seed_set_szie = seed_set_szie
        self.seed_set = []
        self.num_influenced_before = 0
        self.min_actived_nodes = min_actived_nodes

        self.reward_scale = reward_scale

        if self.num_nodes == 0:
            raise Exception(f"Unable to load dataset {dataset} from {edge_file_path}")

        self.sources = []
        self.destinations = []
        self.timestamps = []
        self.labels = []

        for edge in self.edges:
            self.sources.append(edge[0])
            self.destinations.append(edge[1])
            self.timestamps.append(edge[2])
            self.labels.append(1)

        # Compute time statistics
        (
            self.mean_time_shift_src,
            self.std_time_shift_src,
            self.mean_time_shift_dst,
            self.std_time_shift_dst,
        ) = compute_time_statistics(self.sources, self.destinations, self.timestamps)

        self.full_data = Data(
            np.array(self.sources),
            np.array(self.destinations),
            np.array(self.timestamps),
            np.array(range(len(self.edges))),
            np.array(self.labels),
        )

        self.full_reduced_action_space = []
        self.full_reduced_action_space = self.get_full_reduced_action_space()

        self.first_action = None
        self.first_action_influence = None
        if self.first_action is None:
            self.get_first_action()
        self.reset()

    def get_node_reduced_action_space_id(self, nodes: list):
        """Get the nodes' index in the full_reduced_action_space.
        Raises ValueError if any node is not in the reduced action space"""
        index_map = {
            value: index for index, value in enumerate(self.full_reduced_action_space)
        }

        for node in nodes:
            if node not in index_map:
                raise ValueError(f"Node {node} is not in the reduced action space")

        return [index_map[x] for x in nodes]

    def get_full_reduced_action_space(self):
        if self.full_reduced_action_space != []:
            return self.full_reduced_action_space

        full_reduced_action_space = []
        node_activated_nodes = {}

        for edge in self.edges:
            src, des = edge[0], edge[1]
            if src not in node_activated_nodes.keys():
                node_activated_nodes[src] = set()
            if des not in node_activated_nodes.keys():
                node_activated_nodes[des] = set()
            node_activated_nodes[src].add(des)

        for key in node_activated_nodes.keys():
            if len(node_activated_nodes[key]) >= self.min_actived_nodes:
                full_reduced_action_space.append(key)

        return list(np.sort(full_reduced_action_space))

    def get_number_nodes(self):
        return self.num_nodes

    def get_neighbour_finder(self):
        return get_neighbor_finder(self.full_data, False)

    def get_time_statistics(self):
        return [
            self.mean_time_shift_src,
            self.std_time_shift_src,
            self.mean_time_shift_dst,
            self.std_time_shift_dst,
        ]

    def get_edge_list(self):
        return self.edges

    def get_seed_set(self):
        return self.seed_set

    def get_action_space(self):
        return [x for x in self.full_reduced_action_space if x not in self.seed_set]

    def reset(self):
        self.seed_set = [self.first_action]
        self.num_influenced_before = self.first_action_influence

        return copy.deepcopy(self.seed_set)

    def full_reset(self):
        self.seed_set = []
        self.num_influenced_before = 0

        return copy.deepcopy(self.seed_set)

    def step(self, action: int, repeat_times: int = 1000):
        """
        paramaters:
            action: index of node
        returns:
            next_state: [seed set]
            reward
            done: reach the threshold of seed set
        """
        if action >= self.num_nodes:
            raise Exception(
                f"Invalid action as node index {action} exceeds the size of nodes!"
            )

        self.seed_set.append(action)
        num_influenced = self.socialsis.influence_time(
            self.seed_set, repeat_times=repeat_times
        )
        reward = (num_influenced - self.num_influenced_before) * self.reward_scale
        self.num_influenced_before = num_influenced

        if len(self.seed_set) == self.seed_set_szie:
            done = True
        else:
            done = False

        state = copy.deepcopy(self.seed_set)

        return state, reward, done

    def step_no_reward(self, action: int):
        """
        paramaters:
            action: index of node
        returns:
            next_state: [seed set]
            done: reach the threshold of seed set
        """
        if action >= self.num_nodes:
            raise Exception(
                f"Invalid action as node index {action} exceeds the size of nodes!"
            )

        self.seed_set.append(action)

        if len(self.seed_set) == self.seed_set_szie:
            done = True
        else:
            done = False

        state = copy.deepcopy(self.seed_set)

        return state, done

    def seed_set_influence(self, repeat_times=1000):
        return self.socialsis.influence_time(self.seed_set, repeat_times=repeat_times)

    def get_first_action(self, repeat_times=1000):
        # self.first_action = 805
        # self.first_action_influence = self.socialsis.influence_time([805], repeat_times=repeat_times)
        # return
        best_influence = -1
        best_node = -1
        for node in tqdm(
            self.full_reduced_action_space, desc="Finding best first action"
        ):
            node_influence = self.socialsis.influence_time(
                [node], repeat_times=repeat_times
            )
            if node_influence > best_influence:
                best_influence = node_influence
                best_node = node
        self.first_action = best_node
        self.first_action_influence = best_influence
