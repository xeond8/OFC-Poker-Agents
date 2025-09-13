import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from bot.states import StateOneBoard, Action
from bot.environment import assignments, Board, Environment
import pickle
from collections import deque
from tqdm import trange



def int_to_action(ind: int, state: StateOneBoard):
    i = ind % 3
    real_hand = state.env.hand[:i] + state.env.hand[i+1:]
    assignment = assignments[ind // 3]
    cur_move = [[], [], []]

    for it in zip(real_hand, assignment):
        cur_move[it[1]].append(it[0])

    action = Action(state.env.first_player, cur_move[0], cur_move[1], cur_move[2])

    return action



class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13*5 + 3*5 + 3 + 1, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 64)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64, 9)
        self.batch_norm5 = nn.BatchNorm1d(9)
        self.layer6 = nn.Linear(9, 27)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.layer1(x)))
        x = self.activation(self.batch_norm2(self.layer2(x)))
        x = self.activation(self.batch_norm3(self.layer3(x)))
        x = self.activation(self.batch_norm4(self.layer4(x)))
        x = self.activation(self.batch_norm5(self.layer5(x)))
        x = self.layer6(x)
        return x


if __name__ == "__main__":

    learning_rate = 1e-3
    num_episodes = 7200
    neg_reward = -10
    batch_size = 64
    target_update = 100

    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    memory = deque()

    with open('../data/env_move1.pkl', 'rb') as f:
        env_arr = pickle.load(f)


    def select_action(state: StateOneBoard, epsilon=0.1):
        if random.random() < epsilon:
            act_ind = random.sample(range(27), 1)[0]

        else:
            policy_net.eval()
            with torch.no_grad():
                state_vector = torch.FloatTensor(state.flatten()).unsqueeze(0)
                q_values = policy_net(state_vector)
                act_ind = q_values.argmax().item()

        return int_to_action(act_ind, state), act_ind


    def optimize_model():
        policy_net.train()
        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)

        states, actions_nums, states_new, rewards, dones = zip(*batch)

        states = [x.flatten() for x in states]
        states_new = [x.flatten() for x in states_new]

        states = torch.FloatTensor(states)
        actions_nums = torch.LongTensor(actions_nums).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(states_new)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = policy_net(states).gather(1, actions_nums)
        next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
        predict_q = rewards + next_q * (1 - dones)

        loss = loss_fn(current_q, predict_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in trange(num_episodes):
        env = env_arr[episode]
        state = StateOneBoard(env)

        done = False
        reward = 0
        while not done:
            action, action_ind = select_action(state)
            state_new = state.take_action(action)

            if not state.is_correct():
                done = True
                reward += neg_reward

            elif state.is_terminal():
                done = True
                reward += state.get_reward()

            memory.append((state, action_ind, state_new, reward, done))

            optimize_model()

            state = state_new

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

    torch.save(target_net.state_dict(), "dqn_model.pt")







