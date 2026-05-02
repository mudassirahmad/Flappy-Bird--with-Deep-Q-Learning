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
import shutil

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f"Training on: {device}")

class FlappyBirdAgent:

    def __init__(self, hyperparameters_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameters = yaml.safe_load(f)
            hyperparameters = all_hyperparameters[hyperparameters_set]

        self.hyperparameters_set = hyperparameters_set

        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor = hyperparameters["discount_factor_g"]
        self.step_on_reward = hyperparameters["step_on_reward"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        self.env_make_params = hyperparameters.get("env_make_params",{})
        self.enable_double_dqn = hyperparameters["enable_double_dqn"]
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn'] 

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None


        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.png')


    def save_to_kaggle_output(self):
        os.makedirs('/kaggle/output/runs', exist_ok=True)
        # Copy model
        if os.path.exists(self.MODEL_FILE):
            shutil.copy(self.MODEL_FILE, f'/kaggle/output/runs/{self.hyperparameters_set}.pt')
        # Copy graph
        if os.path.exists(self.GRAPH_FILE):
            shutil.copy(self.GRAPH_FILE, f'/kaggle/output/runs/{self.hyperparameters_set}.png')
        # Copy log
        if os.path.exists(self.LOG_FILE):
            shutil.copy(self.LOG_FILE, f'/kaggle/output/runs/{self.hyperparameters_set}.log')
        # Save epsilon value too
        epsilon_file = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}_epsilon.txt')
        if os.path.exists(epsilon_file):
            shutil.copy(epsilon_file, f'/kaggle/output/runs/{self.hyperparameters_set}_epsilon.txt')
    def run(self, is_training=True, render=False): 

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        run_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history = []
        policy = DQN(num_states, run_actions,self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            replay_memory = ExperienceReplay(self.replay_memory_size)
            
            epsilon = self.epsilon_init

            target = DQN(num_states, run_actions,self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target.load_state_dict(policy.state_dict())

            
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate_a)
            step_count = 0

            best_reward = float(-9999999)
            # added check for existing model to resume training from the last model because kaggle sessions can be interrupted after 12 hours
            if os.path.exists(self.MODEL_FILE):
                print("Resuming from saved model...")
                policy.load_state_dict(torch.load(self.MODEL_FILE))
                target.load_state_dict(torch.load(self.MODEL_FILE))

                ############# 
                # ON Your first run make sure to comment out the following lines that load the previous best reward and epsilon value, 
                # otherwise it will resume with the best reward and epsilon value from the previous run which might not be ideal for your first run. 
                # After your first run, you can uncomment these lines to enable resuming with the previous best reward and epsilon value for continued training.
                #############
                
                
                #loading previous best reward for continued model training
                # best_reward_file = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}_best_reward.txt')
                # if os.path.exists(best_reward_file):
                #     with open(best_reward_file, 'r') as f:
                #         best_reward = float(f.read())
                #     print(f"Resuming with best reward: {best_reward}")
                #loading previous epsilon value for continued model training
                # epsilon_file = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}_epsilon.txt')
                # if os.path.exists(epsilon_file):
                #     with open(epsilon_file, 'r') as f:
                #         epsilon = float(f.read())
                #     print(f"Resuming with epsilon: {epsilon}")
                best_reward = 332.6
                epsilon = self.epsilon_min
        else:
            policy.load_state_dict(torch.load(self.MODEL_FILE))
            policy.eval()
            epsilon = self.epsilon_min
        
        for episode in itertools.count():
            state, _ = env.reset()

            if not is_training and episode >= 10:  # runs 10 test episodes then stops
                break

            state= torch.tensor(state, dtype=torch.float32, device=device)
            
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.step_on_reward:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy(state.unsqueeze(dim=0)).squeeze().argmax()
                    
                
                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action.item())
                # print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {episode_reward}")
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                
                if is_training:
                    replay_memory.append((state, action, new_state, reward, terminated))

                    step_count += 1
                
                
                
                #move to the next state
                state = new_state


                
            rewards_per_episode.append(episode_reward.cpu().item() if torch.is_tensor(episode_reward) else episode_reward)
            
            # Store the model when the best reward is achieved
            if is_training:
                if episode_reward > best_reward:
                    improvement = f"{(episode_reward-best_reward)/best_reward*100:+.1f}%" if best_reward != float('-inf') else "first save"
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({improvement}) at episode {episode}"
                    #log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(log_message + "\n")
                    torch.save(policy.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                    
                    # Save epsilon
                    with open(os.path.join(RUNS_DIR, f'{self.hyperparameters_set}_epsilon.txt'), 'w') as f:
                        f.write(str(epsilon))
                    # Save best reward
                    with open(os.path.join(RUNS_DIR, f'{self.hyperparameters_set}_best_reward.txt'), 'w') as f:
                        f.write(str(best_reward))

                    self.save_to_kaggle_output()

                current_time = datetime.now()

                if current_time - last_graph_update_time >= timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time



                if len(replay_memory) >= self.mini_batch_size:
                    mini_batch = replay_memory.sample(self.mini_batch_size)
                    self.optimize_model(mini_batch, policy, target)

                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                    epsilon_history.append(epsilon)

                    if step_count >= self.network_sync_rate:
                        target.load_state_dict(policy.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig=plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean([r.cpu().item() if torch.is_tensor(r) else r 
                           for r in rewards_per_episode[max(0, x-99):(x+1)]])

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
        self.save_to_kaggle_output()


    def optimize_model(self, mini_batch, policy, target):
        state, action, new_state, reward, terminated = zip(*mini_batch)
            
        state = torch.stack(state)
        
        action = torch.stack(action)

        new_state = torch.stack(new_state)

        reward = torch.stack(reward)

        terminated =  torch.tensor(terminated).float().to(device)
        #double dqn: use policy to select the best action, but use target to calculate the q value of that action. This way we avoid overestimation bias.
        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy(new_state).argmax(1)
                target_q_value = reward + (1 - terminated) * self.discount_factor * target(new_state).gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()
            
            else:
                target_q_value = reward + (1 - terminated) * self.discount_factor * target(new_state).max(dim=1)[0]

        current_q_value = policy(state).gather(dim=1, index=action.unsqueeze(dim=1)).squeeze()
        
        
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

    dql = FlappyBirdAgent(hyperparameters_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
    