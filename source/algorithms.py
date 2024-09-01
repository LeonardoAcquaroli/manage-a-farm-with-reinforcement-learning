import torch
import gymnasium as gym
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import PolynomialFeatures

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
    
    def policy(self, state: int) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        options = self.env.unwrapped.actions_available
        if np.random.random() < self.epsilon:
            return np.random.choice(options)
        else:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return options[np.argmax(available_values)]

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
        
    def policy(self, state: dict) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        options = self.env.unwrapped.actions_available
        if np.random.random() < self.epsilon:
            return np.random.choice(options)
        else:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return options[np.argmax(available_values)]
        
    def greedy_policy(self, state: dict) -> int:
        """Implements greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        options = self.env.unwrapped.actions_available
        available_values = [self.q(state, a).detach().numpy() for a in options]
        return options[np.argmax(available_values)]

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

import torch
import gymnasium as gym
import numpy as np
from collections import defaultdict

class FarmAgentREINFORCEAdvantage:
    """REINFORCE with Advantage"""
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
        self.policy_w = torch.tensor(i_policy_w, dtype=float, requires_grad=True)
        
        if initial_value_w is None:
            i_value_w = np.ones(self.env.unwrapped.features_number)
        else:
            i_value_w = initial_value_w
        self.value_w = torch.tensor(i_value_w, dtype=float, requires_grad=True)
        
        self.policy_optimizer = torch.optim.Adam([self.policy_w], lr=self.policy_alpha)
        self.value_optimizer = torch.optim.Adam([self.value_w], lr=self.value_alpha)
    
    def x(self, state: dict) -> torch.Tensor:
        """Returns the features that represent the state"""
        features = np.array(list(state.values()))
        return torch.tensor(features, dtype=float)
    
    def policy(self, state: dict) -> int:
        """Implements e-greedy strategy for action selection"""
        options = self.env.unwrapped.actions_available
        if np.random.random() < self.epsilon:
            return np.random.choice(options)
        else:
            state_features = self.x(state)
            action_probs = torch.softmax(self.policy_w @ state_features, dim=0)
            return np.random.choice(options, p=action_probs.detach().numpy())
    
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
        returns = torch.tensor(returns)
        
        # Calculate state values
        state_values = torch.tensor([self.value(s) for s in states])
        
        # Calculate advantages
        advantages = returns - state_values
        
        # Update policy
        for t, (state, action) in enumerate(zip(states, actions)):
            state_features = self.x(state)
            action_probs = torch.softmax(self.policy_w @ state_features, dim=0)
            log_prob = torch.log(action_probs[action])
            policy_loss = -log_prob * advantages[t]
            
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