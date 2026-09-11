import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from datetime import datetime

from Env.IMEnv import IMEnv
from Model.rldim import RLDIM

import argparse
import copy


class ReplayMemory:
    def __init__(self, memory_size) -> None:
        self.memory = deque(maxlen=memory_size)

    def push(self, transition) -> None:
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def get_length(self):
        return len(self.memory)


class DoubleDQN(nn.Module):
    def __init__(
        self,
        memory_dimension,
        time_encoder_dimension,
        device,
        number_nodes,
        number_embedding_layer=1,
        batch_size=200,
    ):
        super(DoubleDQN, self).__init__()
        self.memory_dimension = memory_dimension
        self.time_encoder_dimension = time_encoder_dimension
        self.device = device
        self.number_embedding_layer = number_embedding_layer
        self.batch_size = batch_size
        self.updated = True
        self.rldim = RLDIM(
            memory_dimension=memory_dimension,
            time_encoder_dimension=time_encoder_dimension,
            device=device,
            number_nodes=number_nodes,
        )

    """
    Set the update state to true to make the model recalculate embedding
    Update state will be set to false once the embedding is recalculated
    """

    def set_update_state(self, update_state):
        self.updated = update_state

    def ensure_embedding(self, env):
        """Compute and cache node embeddings (no-op if already fresh)."""
        if not self.updated:
            return
        edge_list = copy.deepcopy(env.get_edge_list())
        scale = edge_list[-1][-1] / 10.0
        for e in edge_list:
            e[-1] /= scale
        n = env.get_number_nodes()
        mem = self.rldim.computer_memory(edge_list, self.batch_size, n)
        self.nodes_embedding = self.rldim.compute_embedding(
            mem, env.get_neighbour_finder(), final_timestamp=edge_list[-1][-1]
        )
        self.graph_representation = self.rldim.compute_graph_representation(
            self.nodes_embedding
        )
        self.updated = False

    def forward(self, state: list, actions: list, env):
        """
        Parameters:
            state: seed set
            actions: index of the nodes that the actions will add into
            env: object of class IMEnv
            batch_size: batch size to generate the memory
        Return:
            tensor: value of the node
        """

        edge_list = copy.deepcopy(env.get_edge_list())
        largest_timestamp = edge_list[-1][-1]
        scale_factor = largest_timestamp / 10.0
        for edge in edge_list:
            edge[-1] /= scale_factor
        number_nodes = env.get_number_nodes()
        if self.updated:
            nodes_memory = self.rldim.computer_memory(
                edge_list=edge_list,
                batch_size=self.batch_size,
                number_nodes=number_nodes,
            )
            self.nodes_embedding = self.rldim.compute_embedding(
                nodes_memory,
                env.get_neighbour_finder(),
                final_timestamp=edge_list[-1][-1],
            )
            self.graph_representation = self.rldim.compute_graph_representation(
                self.nodes_embedding
            )
            self.updated = False

        after_nodes_value = self.rldim.calculate_value(
            self.nodes_embedding,
            self.graph_representation,
            state,
            actions,
        )
        return after_nodes_value


class Agent:
    def __init__(
        self,
        memory_dimension,
        time_encoder_dimension,
        device,
        number_nodes,
        replay_memory_size=1000,
        number_embedding_layer=1,
        batch_size_RL=200,
        batch_size_GE=200,
        gamma=0.95,
        epsilon_start=1,
        epsilon_end=0.1,
        epsilon_decay=0.95,
        lr=1e-3,
        optimizer="Adam",
        grad_clip_value=0.1,
    ):
        self.batch_size_RL = batch_size_RL
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.grad_clip_value = grad_clip_value
        self.q_network = DoubleDQN(
            memory_dimension=memory_dimension,
            time_encoder_dimension=time_encoder_dimension,
            number_embedding_layer=number_embedding_layer,
            device=self.device,
            number_nodes=number_nodes,
            batch_size=batch_size_GE,
        ).to(self.device)
        self.target_network = DoubleDQN(
            memory_dimension=memory_dimension,
            time_encoder_dimension=time_encoder_dimension,
            number_embedding_layer=number_embedding_layer,
            device=self.device,
            number_nodes=number_nodes,
            batch_size=batch_size_GE,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        if optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr)

    def soft_update_target_network(self, tau=0.01):

        for target_param, q_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            # Target = tau × Q_new + (1-tau) × Target_old
            target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)

        self.target_network.set_update_state(update_state=True)

    def choose_action(self, env):
        random_state = False
        action_space = env.get_action_space()
        #  choose action randomly
        if np.random.rand() < self.epsilon:
            random_state = True
            action = random.choice(action_space)
        #  choose action according to q values
        else:
            with torch.no_grad():
                nodes_value = self.q_network(env.get_seed_set(), action_space, env)
            max_index = int(torch.argmax(nodes_value))
            action = action_space[max_index]
        return action, random_state

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon * self.epsilon_decay

    def learn(self, env):
        if self.replay_memory.get_length() < self.batch_size_RL:
            return False

        minibatch = self.replay_memory.sample(self.batch_size_RL)

        states = [t[0] for t in minibatch]
        actions = [t[1] for t in minibatch]
        rewards = torch.tensor(
            [t[2] for t in minibatch], dtype=torch.float32, device=self.device
        )
        next_states = [t[3] for t in minibatch]
        dones = torch.tensor(
            [float(t[4]) for t in minibatch], dtype=torch.float32, device=self.device
        )

        full_space = env.get_full_reduced_action_space()

        self.q_network.set_update_state(True)
        self.q_network.ensure_embedding(env)
        q_emb, q_graph = (
            self.q_network.nodes_embedding,
            self.q_network.graph_representation,
        )

        q_values = self.q_network.rldim.calculate_value_batch(
            q_emb, q_graph, states, actions
        )

        next_actions = []
        with torch.no_grad():
            for ns in next_states:
                as_i = [x for x in full_space if x not in ns]
                vals = self.q_network.rldim.calculate_value(q_emb, q_graph, ns, as_i)
                next_actions.append(as_i[int(torch.argmax(vals))])

        self.target_network.set_update_state(True)
        self.target_network.ensure_embedding(env)
        tgt_emb = self.target_network.nodes_embedding
        tgt_graph = self.target_network.graph_representation

        with torch.no_grad():
            expected_next_q = self.target_network.rldim.calculate_value_batch(
                tgt_emb, tgt_graph, next_states, next_actions
            )

        expected_q = rewards + (1 - dones) * self.gamma * expected_next_q
        loss = nn.functional.smooth_l1_loss(q_values, expected_q.detach())

        write_log(f"loss: {loss.item():.5f}")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), max_norm=1.0
        )
        write_log(f"gradient norm: {grad_norm:.5f}")
        self.optimizer.step()
        self.q_network.set_update_state(True)
        return True

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename):
        state_dict = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
        self.target_network.set_update_state(update_state=True)
        self.q_network.set_update_state(update_state=True)


def train(env, agent, save_model_name):
    bast_eposide_score = 0
    best_eposide_index = 0
    no_improve = 0

    q_update_time = 0

    start_time = datetime.now()

    for i_episode in range(2000):
        episode_start_time = datetime.now()

        state = env.reset()

        is_learn = False

        for action_index in range(1000):
            action, action_random = agent.choose_action(env)
            next_state, reward, done = env.step(action)
            agent.replay_memory.push((state, action, reward, next_state, done))
            state = next_state
            if action_random:
                write_log(
                    "random action: {:>6} | reward: {:>8.5f}".format(action, reward)
                )
            else:
                write_log(
                    "action       : {:>6} | reward: {:>8.5f}".format(action, reward)
                )

            if (
                agent.replay_memory.get_length() > 2 * BATCH_SIZE_RL
                and (action_index + 2) % 10 == 0
            ):
                is_learn = agent.learn(env)
                if is_learn:
                    q_update_time += 1
                    write_log(f"q-network updated for {q_update_time} times")

                    agent.soft_update_target_network()

                    temp_seedset = env.seed_set
                    temp_num_influenced_before = env.num_influenced_before

                    # evaluate the episode using the model without random action
                    eposide_evaluation_score = agent_evaluation(env=env, agent=agent)
                    write_log("Evaluation: {:.5f}".format(eposide_evaluation_score))

                    env.seed_set = temp_seedset
                    env.num_influenced_before = temp_num_influenced_before

                    if eposide_evaluation_score > bast_eposide_score:
                        bast_eposide_score = eposide_evaluation_score
                        best_eposide_index = i_episode + 1
                        no_improve = 0
                        # save_model_path = f"saved_model/{save_model_name}.pth"
                        # torch.save(agent.q_network.state_dict(), save_model_path)
                    else:
                        no_improve += 1
                        if no_improve > 200 and q_update_time > 200:
                            return best_eposide_index, bast_eposide_score

                    agent.update_epsilon()

            if done:
                break

        current_time = datetime.now()
        episode_runnming_time = current_time - episode_start_time
        total_runnming_time = current_time - start_time
        write_log(
            f"Episode {i_episode+1}: runnming time: {episode_runnming_time}, total running time: {total_runnming_time}"
        )

    return best_eposide_index, bast_eposide_score


def agent_evaluation(env, agent):
    #  Do the evaluation with the model (no random action)
    current_epsilon = agent.epsilon
    agent.epsilon = 0
    env.reset()
    for _ in range(100):
        action, _ = agent.choose_action(env)
        _, done = env.step_no_reward(action)
        if done:
            break
    write_log(f"seed set: {env.get_seed_set()}")
    episode_score = env.seed_set_influence()
    agent.epsilon = current_epsilon
    return episode_score


def write_log(content):
    if run_mode == "train":
        content += "\n"
        with open(log_file, "a") as file:
            file.write(content)
    else:
        print(content)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--activation_length", type=int)
parser.add_argument("--size_seed_set", type=int)
parser.add_argument("--minimal_activated_nodes", type=int)
parser.add_argument("--reward_scale", type=float)
parser.add_argument("--edge_thresholds", type=float, default=0.5)
args = parser.parse_args()

DATASET = args.dataset
ACTIVATION_LENGTH = args.activation_length  # in days
REPLAY_MEMORY_SIZE = 50000
LERNING_RATE = 7e-4
EPSILON_START = 1
EPSILON_END = 0.4
EPSILON_DECAY = 0.995
GAMMA = 0.95
MEMORY_SIZE = 63
TIME_ENCODING_SIZE = 1
BATCH_SIZE_RL = 256
BATCH_SIZE_GE = 200
OPTIMIZER = "Adam"
SEED_SET_SIZE = args.size_seed_set
MINIMAL_ACTIVATED_NODES = args.minimal_activated_nodes
REWARD_SCALE = args.reward_scale
EDGE_THRESHOLD = args.edge_thresholds

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
run_mode = "train"
# run_mode = "debug"

start_time = datetime.now().strftime("%Y%m%d%H%M%S")
random_file_id = random.randint(0, 9)
log_file = f"./log/{DATASET}_{start_time}_{random_file_id}.txt"

write_log(
    f"Dataset: {DATASET}, seed set size: {SEED_SET_SIZE}, activation length: {ACTIVATION_LENGTH} days"
)
write_log(
    f"memory size: {MEMORY_SIZE} time encoding size: {TIME_ENCODING_SIZE}, batch size(GE): {BATCH_SIZE_GE}, batch size(RL): {BATCH_SIZE_RL}"
)
write_log(f"optimizer: {OPTIMIZER}")
write_log(f"LR: {LERNING_RATE}, gamma: {GAMMA}")
write_log(
    f"update q-network every 10 steps, soft update target-network every step with tau=0.01"
)
write_log(
    f"dynamic epsilon, epsilon stars from {EPSILON_START} to {EPSILON_END} with decay rate {EPSILON_DECAY}, decay every episode"
)
write_log(
    f"Nodes in the reduced action space: Activate minimal {MINIMAL_ACTIVATED_NODES} nodes"
)

env = IMEnv(
    dataset=DATASET,
    edge_thresholds=0.5,
    activation_length_percent=0,
    activation_length=ACTIVATION_LENGTH * 86400,
    seed_set_szie=SEED_SET_SIZE,
    min_actived_nodes=MINIMAL_ACTIVATED_NODES,
    reward_scale=REWARD_SCALE,
)


write_log(
    f"Size of the full reduced action space {len(env.get_full_reduced_action_space())}"
)

agent = Agent(
    memory_dimension=MEMORY_SIZE,
    time_encoder_dimension=TIME_ENCODING_SIZE,
    device=DEVICE,
    number_nodes=env.get_number_nodes(),
    replay_memory_size=REPLAY_MEMORY_SIZE,
    batch_size_RL=BATCH_SIZE_RL,
    batch_size_GE=BATCH_SIZE_GE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    lr=LERNING_RATE,
    optimizer=OPTIMIZER,
)

best_eposide_index, bast_eposide_score = train(
    env=env,
    agent=agent,
    save_model_name=f"{DATASET}_{start_time}_{random_file_id}",
)

write_log(f"Best eposide: {best_eposide_index}, score: {bast_eposide_score}")
