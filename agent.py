import torch
from torch import nn
from dqn import DQN
import flappy_bird_gymnasium
import gymnasium
from experience_replay import ExperienceReplay
import itertools
import yaml
import random

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

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True, render=False): 

        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        run_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history = []
        policy = DQN(num_states, run_actions).to(device)

        if is_training:
            replay_memory = ExperienceReplay(maxLen=10000, seed=42)
            
            epsilon = self.epsilon_init

            target = DQN(num_states, run_actions).to(device)
            target.load_state_dict(policy.state_dict())

            
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate_a)
            step_count = 0
        
        for episode in itertools.count():
            state, _ = env.reset()
            state= torch.tensor(state, dtype=torch.float32, device=device)
            
            terminated = False
            episode_reward = 0.0
            while not terminated:

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
                # Store the transition in replay memory
                if is_training:
                    replay_memory.append((state, action, reward, new_state, terminated))
                    step_count += 1 
                
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

    def optimize_model(self, mini_batch, policy, target):
        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target_q_value = reward
            else:             
                with torch.no_grad():
                    target_q_value = reward + self.discount_factor * target(new_state).max()
            
            current_q_value = policy(state)
            # Compute the loss
            loss = self.loss_fn(current_q_value, target_q_value)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



if __name__ == "__main__":
    agent = FlappyBirdAgent("cartpole1")
    agent.run(is_training=True, render=True)