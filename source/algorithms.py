import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
import numpy as np
from collections import defaultdict
# from sklearn.preprocessing import PolynomialFeatures

class FarmAgentSarsaVFA:
    """Sarsa VFA"""
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95, initial_w: np.ndarray = None) -> None:
        self.env = environment
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        # self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.training_error = []
        if initial_w is None:
            i_w = np.ones((self.env.action_space.n, self.env.unwrapped.features_number))
        else:
            i_w = initial_w
        self.w = torch.tensor(i_w, dtype=float, requires_grad=True)
    
    def x(self, state: dict) -> np.ndarray:
        """Returns the features that represents the state

        Args:
            state (dict): The state as a dictionary obtained as the observation object returned by a step
        Returns:
            torch.tensor: state observed by the agent in terms of state features 
        """
        features = np.array(list(state.values()))
        # features = self.poly.fit_transform(features.reshape(1, -1)).flatten()
        return torch.tensor(features, dtype=float, requires_grad=False)
    
    def q(self, state: int, action: int) -> float:
        return self.x(state) @ self.w[action, :]
    
    def policy(self, state: int, greedy) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        options = self.env.unwrapped.actions_available
        if (np.random.random() > self.epsilon) or greedy:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return options[np.argmax(available_values)]
        else:
            return np.random.choice(options)

    def update(self, state: int, action: int, reward: float, s_prime: int):
        next_action = self.policy(state)
        q_target = reward + self.gamma * self.q(s_prime, next_action) 
        q_value = self.q(state, action)
        q_error = q_target - q_value
        q_value.backward()
        with torch.no_grad():
            self.w += self.alpha * q_error * self.w.grad 
            self.w.grad.zero_()
        self.training_error.append(q_error.detach().numpy())
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

class FarmAgentMCVFA:
    """Montecarlo VFA"""
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95, initial_w: np.ndarray = None) -> None:
        self.env = environment
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        # self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.training_error = []
        self.episode_actions = {}
        if initial_w is None:
            i_w = np.ones((self.env.action_space.n, self.env.unwrapped.features_number))
        else:
            i_w = initial_w
        self.w = torch.tensor(i_w, dtype=float, requires_grad=True)
        self.R = defaultdict(list)
    
    def x(self, state: dict) -> np.ndarray:
        """Returns the features that represents the state

        Args:
            state (dict): The state as a dictionary obtained as the observation object returned by a step
        Returns:
            torch.tensor: state observed by the agent in terms of state features 
        """
        features = np.array(list(state.values()))
        # features = self.poly.fit_transform(features.reshape(1, -1)).flatten()
        return torch.tensor(features, dtype=float, requires_grad=False)
    
    def q(self, state: int, action: int) -> float:
        return self.x(state) @ self.w[action, :]
        
    def policy(self, state: int, greedy) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        options = self.env.unwrapped.actions_available
        if (np.random.random() > self.epsilon) or greedy:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return options[np.argmax(available_values)]
        else:
            return np.random.choice(options)

    def generate_episode(self, max_iterations: int = 30):
        e = []
        state, info = self.env.reset()
        for i in range(max_iterations):
            action = self.policy(state)
            s_prime, reward, terminated, truncated, info = self.env.step(action=action)
            state_as_frozenset = frozenset(state.items()) # Convert the dictionary to a frozenset of items to avoid unashable type error
            e.append((state_as_frozenset, action, (self.gamma ** (i+1)) * reward))
            if terminated or truncated:
                break
            else:
                state = s_prime
        return e

    def update(self,  episode_number: int, max_iterations: int = 30):
        """MC update rule"""
        # Generate an episode
        episode = self.generate_episode(max_iterations=max_iterations)
        visited = set()
        for i, (s, a, _) in enumerate(episode):
            # Statistics
            # Store the actions taken in the episode
            if episode_number not in self.episode_actions.keys():
                self.episode_actions[episode_number] = []
            self.episode_actions[episode_number].append(a)

            # Find G (substitute of the ground truth) if first visit of (s,a)
            if (s, a) not in visited:
                visited.add((s, a))
                G = sum(rw for _, _, rw in episode[i:])
                self.R[(s, a)].append(G)
        for state, action in visited:
            q_target = np.mean(self.R[(state, action)])
            state = dict(state)
            q_value = self.q(state, action)
            q_error = q_target - q_value
            q_value.backward()
            with torch.no_grad():
                self.w += self.alpha * q_error * self.w.grad 
                self.w.grad.zero_()
            self.training_error.append(q_error.detach().numpy())
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

class FarmAgentREINFORCEAdvantage:
    def __init__(self, environment: gym.Env, policy_learning_rate: float, value_learning_rate: float,
                 epsilon: float, epsilon_decay: float, final_epsilon: float,
                 gamma: float = 0.99, initial_policy_w: np.ndarray = None, initial_value_w: np.ndarray = None) -> None:
        self.env = environment
        self.policy_alpha = policy_learning_rate
        self.value_alpha = value_learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        
        if initial_policy_w is None:
            i_policy_w = np.ones((self.env.action_space.n, self.env.unwrapped.features_number))
        else:
            i_policy_w = initial_policy_w
        self.policy_w = torch.tensor(i_policy_w, dtype=torch.float32, requires_grad=True)
        
        if initial_value_w is None:
            i_value_w = np.ones(self.env.unwrapped.features_number)
        else:
            i_value_w = initial_value_w
        self.value_w = torch.tensor(i_value_w, dtype=torch.float32, requires_grad=True)
        
        self.policy_optimizer = torch.optim.Adam([self.policy_w], lr=self.policy_alpha)
        self.value_optimizer = torch.optim.Adam([self.value_w], lr=self.value_alpha)
    
    def x(self, state: dict) -> torch.Tensor:
        """Returns the features that represent the state"""
        features = np.array(list(state.values()), dtype=np.float32)
        return torch.tensor(features, dtype=torch.float32, requires_grad=True)
    
    def policy(self, state):
        """Implements e-greedy strategy for action selection"""
        options = self.env.unwrapped.actions_available
        
        if np.random.random() < self.epsilon:
            return np.random.choice(options)
        else:
            state_features = self.x(state)
            # Only consider the rows of policy_w corresponding to available actions
            available_action_weights = self.policy_w[options, :]
            action_scores = available_action_weights @ state_features
            action_probs = torch.softmax(action_scores, dim=0)
            print(f'''WEIGHTS: {available_action_weights}\n
                  state_features: {state_features}\n
                  options: {options}\n
                  action_scores: {action_scores}\n
                  action_probs: {action_probs}''')
            return options[np.random.choice(len(options), p=action_probs.detach().numpy())]
    
    def value(self, state: dict) -> float:
        """Estimates the value of a state"""
        state_features = self.x(state)
        return torch.dot(self.value_w, state_features)
    
    def generate_episode(self, max_iterations: int = 30):
        episode = []
        state, info = self.env.reset()
        for _ in range(max_iterations):
            action = self.policy(state)
            s_prime, reward, terminated, truncated, info = self.env.step(action=action)
            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = s_prime
        return episode
    
    def update(self, episode_number: int, max_iterations: int = 30):
        """REINFORCE with Advantage update rule"""
        episode = self.generate_episode(max_iterations=max_iterations)
        
        states, actions, rewards = zip(*episode)
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, requires_grad=True)
        
        # Calculate state values
        state_values = torch.stack([self.value(s) for s in states])
        
        # Calculate advantages
        advantages = returns - state_values.detach()  # Detach state_values to avoid computing gradients through it
        
        # Update policy
        policy_loss = 0
        for t, (state, action) in enumerate(zip(states, actions)):
            state_features = self.x(state)
            action_probs = torch.softmax(self.policy_w @ state_features, dim=0)
            log_prob = torch.log(action_probs[action])
            policy_loss += -log_prob * advantages[t]
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function
        value_loss = torch.mean((returns - state_values) ** 2)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        self.training_error.append(value_loss.item())
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

class FarmAgentNeuralREINFORCEAdvantage:
    def __init__(self, environment, policy_learning_rate, value_learning_rate,
                 epsilon, epsilon_decay, final_epsilon, gamma=0.99):
        self.env = environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        initial_state, _ = self.env.reset()
        self.state_keys = list(initial_state.keys())
        self.state_dim = len(self.state_keys)
        self.action_dim = self.env.action_space.n

        # Initialize policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize optimizers with weight decay for L2 regularization
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_learning_rate, weight_decay=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_learning_rate, weight_decay=1e-5)

        # Learning rate schedulers
        self.policy_scheduler = StepLR(self.policy_optimizer, step_size=5000, gamma=0.95)
        self.value_scheduler = StepLR(self.value_optimizer, step_size=5000, gamma=0.95)

        # Initialize feature normalization parameters
        self.feature_means = {k: 0 for k in self.state_keys}
        self.feature_stds = {k: 1 for k in self.state_keys}
        self.feature_count = 0

    def normalize_state(self, state):
        self.feature_count += 1
        normalized_state = {}
        for k in self.state_keys:
            delta = state[k] - self.feature_means[k]
            self.feature_means[k] += delta / self.feature_count
            delta2 = state[k] - self.feature_means[k]
            self.feature_stds[k] += delta * delta2

            if self.feature_count > 1:
                normalized_state[k] = (state[k] - self.feature_means[k]) / (np.sqrt(self.feature_stds[k] / (self.feature_count - 1)) + 1e-8)
            else:
                normalized_state[k] = state[k]
        
        return np.array([normalized_state[k] for k in self.state_keys])

    def policy(self, state, greedy=False):
        normalized_state = self.normalize_state(state)
        state_tensor = torch.FloatTensor(normalized_state)
        
        options = self.env.unwrapped.actions_available
        if (np.random.random() > self.epsilon) or greedy:
            with torch.no_grad():
                action_scores = self.policy_net(state_tensor)
                action_probs = F.softmax(action_scores, dim=0)
            return options[torch.argmax(action_probs[options]).item()]
        else:
            return np.random.choice(options)

    def update(self, episode_number, max_iterations=30):
        episode = self.generate_episode(max_iterations)
        states, actions, rewards = zip(*episode)
        
        normalized_states = torch.FloatTensor([self.normalize_state(s) for s in states])
        returns = self.compute_returns(rewards)
        
        # Compute value estimates
        values = self.value_net(normalized_states).squeeze()
        
        # Compute advantages with normalization
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        action_probs = F.softmax(self.policy_net(normalized_states), dim=1)
        selected_probs = action_probs[range(len(actions)), actions]
        policy_loss = -(torch.log(selected_probs + 1e-8) * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        # Update value function
        value_loss = F.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.value_optimizer.step()

        # Step the learning rate schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()

        # Check for NaN values
        if torch.isnan(self.policy_net[0].weight).any() or torch.isnan(self.value_net[0].weight).any():
            print(f"NaN detected in weights at episode {episode_number}")
            self.reset_weights()

    def generate_episode(self, max_iterations):
        episode = []
        state, _ = self.env.reset()
        for _ in range(max_iterations):
            action = self.policy(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def reset_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.policy_net.apply(init_weights)
        self.value_net.apply(init_weights)
        print("Weights have been reset due to NaN values.")