import torch
from torch import nn
import numpy as np
from dqn import DQN
import flappy_bird_gymnasium
import gymnasium
from experience_replay import ExperienceReplay
import itertools
import yaml
import random
from datetime import datetime, timedelta
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class FlappyBirdAgent:

    def __init__(self, hyperparameters_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameters = yaml.safe_load(f)
            hyperparameters = all_hyperparameters[hyperparameters_set]

        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor = hyperparameters["discount_factor_g"]
        self.step_on_reward = hyperparameters["step_on_reward"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        self.env_make_params = hyperparameters.get("env_make_params",{})



        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None


        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False): 

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        run_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history = []
        policy = DQN(num_states, run_actions, self.fc1_nodes).to(device)

        if is_training:
            replay_memory = ExperienceReplay(maxLen=10000, seed=42)
            
            epsilon = self.epsilon_init

            target = DQN(num_states, run_actions, self.fc1_nodes).to(device)
            target.load_state_dict(policy.state_dict())

            
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate_a)
            step_count = 0

            best_reward = float('-inf')
        else:
            policy.load_state_dict(torch.load(self.MODEL_FILE))
            policy.eval()
        
        for episode in itertools.count():
            state, _ = env.reset()
            state= torch.tensor(state, dtype=torch.float32, device=device)
            
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.step_on_reward:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy(state.unsqueeze(0)).squeeze().argmax()
                    
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                episode_reward += reward
                # Store the model when the best reward is achieved
                if is_training:
                    if episode_reward > best_reward:
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                        print(log_message)
                        with open(self.LOG_FILE, "a") as log_file:
                            log_file.write(log_message + "\n")
                        torch.save(policy.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                    current_time = datetime.now()
                    if current_time - last_graph_update_time >= timedelta(seconds=0):
                        self.save_graph(rewards_per_episode, epsilon_history)
                        last_graph_update_time = current_time
                
                #move to the next state
                state = new_state


                
            rewards_per_episode.append(episode_reward)

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)


            if len(replay_memory) >= self.mini_batch_size and is_training:
                
                mini_batch = replay_memory.sample(self.mini_batch_size)
                self.optimize_model(mini_batch, policy, target)

                if step_count < self.network_sync_rate:
                    target.load_state_dict(policy.state_dict())
                    step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig=plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        plt.subplot(  1, 2, 1)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)
        
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history, label='Epsilon', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon Decay')

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize_model(self, mini_batch, policy, target):
        state, action, new_state, reward, terminated = zip(mini_batch)
            
        state = torch.stack(state)
        
        action = torch.stack(action)

        new_state = torch.stack(new_state)

        reward = torch.stack(reward)

        terminated =  torch.tensor(terminated).float().to(device)

        with torch.no_grad():
            target_q_value = reward + (1 - terminated) * self.discount_factor * target(new_state).max(1)[0]

            current_q_value = policy(state).gather(1, action.unsqueeze(1)).squeeze()
        
        
        # Compute the loss
        loss = self.loss_fn(current_q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = FlappyBirdAgent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
    agent = FlappyBirdAgent("cartpole1")
    agent.run(is_training=True, render=True)