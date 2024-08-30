import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from typing import Literal
import numpy as np
import math

class FarmEnv(gym.Env):
    def __init__(self, initial_budget=2, sheep_cost=1, wheat_cost=0.02,
                 wool_price=0.01, wheat_price=0.05,
                 max_years=30, wool_fixed_cost=0.009,
                 storm_probability=0.1, incest_penalty: Literal[2,3,4,5] = 3):
        super(FarmEnv, self).__init__()

        # Environment parameters
        self.initial_budget = initial_budget
        self.sheep_cost = sheep_cost
        self.wheat_cost = wheat_cost
        self.wool_price = wool_price
        self.wheat_price = wheat_price
        self.max_years = max_years
        self.wool_fixed_cost = wool_fixed_cost
        self.storm_probability = storm_probability
        self.bought_sheep_count = 0
        self.incest_penalty = incest_penalty
        self.sheep_reproduction_probability = 0


        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: buy sheep, 1: grow wheat, 2: do not invest
        self.observation_space = spaces.Dict({
            "budget": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32), # Assuming a max budget of 1 million (unit is thousands so 1000*1k = 1M)
            "sheep_count": spaces.Discrete(100),  # Assuming a max of 100 sheep
            "bought_sheep_count": spaces.Discrete(100),  # Assuming a max of 100 sheep
            "sheep_reproduction_probability": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "year": spaces.Discrete(self.max_years + 1)  # 0 to max_years
        })

    @property
    def actions_available(self):
        '''
        Returns the available actions based on the current state
        '''
        max_budget = self.observation_space['budget'].high
        max_sheep_n = self.observation_space['sheep_count'].n
        max_bought_sheep_n = self.observation_space['bought_sheep_count'].n
        max_year = self.observation_space['year'].n
        assert (self.budget <= max_budget) and (self.year <= max_year), 'Invalid state'
        
        if (self.budget >= self.sheep_cost) and (self.sheep_count < max_sheep_n) and (self.bought_sheep_count < max_bought_sheep_n):
            return [0, 1, 2]
        elif self.budget < self.wheat_cost:
            return [2]
        else:
            return [1, 2]

    @property
    def features_number(self):
        n_features = len(self.observation_space.keys())
        # n_features_expanded = sum(math.comb(n_features + k - 1, k) for k in range(2 + 1)) # Degree 2 polynomial features expansion 
        # return n_features_expanded
        return n_features    

    def gaussian_reward(delta_budget, year, sigma=9): # sigma could be a function of years too (maybe with some regulations params not to narrow down too much the shape after year 9)
        gaussian_modifier = math.exp(-(year - 30)**2 / (2 * sigma**2))
        return delta_budget * gaussian_modifier

    def reset(self, seed=None, options: dict = {}):
        super().reset(seed=seed)
        self.budget = self.initial_budget
        self.sheep_count = 0
        self.bought_sheep_count = 0
        self.year = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Set wheat_income to 0 in case Buy sheep is selected
        wheat_income = 0
        # Assign bidget before the step to the variable budget_t to compute the delta
        budget_t = self.budget

        if action == 0:  # Buy sheep
            self.budget -= self.sheep_cost
            self.sheep_count += 1
            self.bought_sheep_count += 1
        
        elif action == 1:  # Grow wheat
            self.budget -= self.wheat_cost
            # Calculate wheat income
            wheat_income = 0 if np.random.random() < self.storm_probability else self.wheat_price

        elif action == 2:  # Do not invest
            pass

        # Calculate wool income taking into account the fixed cost
        wool_fixed_cost = self.wool_fixed_cost if self.sheep_count > 0 else 0
        wool_income = self.sheep_count * self.wool_price - wool_fixed_cost
        # Update budget
        self.budget += wool_income + wheat_income
        
        # Sheep reproduction
        if self.sheep_count > 1:
            # Use a modified, polynomial function to calculate the probability of sheep reproduction
            bought_sheep_ratio = self.bought_sheep_count / self.sheep_count
            self.sheep_reproduction_probability = bought_sheep_ratio ** self.incest_penalty # x^penalty
            # Number of new sheep born as a binomial B(n, p) where n=sheep_pairs_number and p=reproduction probability
            sheep_pairs_number = self.sheep_count * (self.sheep_count - 1) // 2
            new_sheeps = np.random.binomial(sheep_pairs_number, self.sheep_reproduction_probability)
            self.sheep_count += new_sheeps

        # Advance year
        self.year += 1
        # Check end conditions
        done = self.budget <= 0 or self.year >= self.max_years # Budget <= 0 impossible, but here in case of further improvements

        observation = self._get_obs()
        reward = self.calculate_reward(delta_budget=(self.budget - budget_t),
                                       year=self.year,
                                       sigma=9)
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def _get_obs(self):
        return {
            "budget": self.budget,
            "sheep_count": self.sheep_count,
            "bought_sheep_count": self.bought_sheep_count,
            "sheep_reproduction_probability": self.sheep_reproduction_probability,
            "year": self.year
        }

register(
    id="FarmEnv-v0",
    entry_point=lambda initial_budget, sheep_cost, wheat_cost,
                        wool_price, wheat_price,
                        max_years, wool_fixed_cost,
                        storm_probability, incest_penalty:
                    FarmEnv(initial_budget,
                            sheep_cost,
                            wheat_cost,
                            wool_price,
                            wheat_price,
                            max_years,
                            wool_fixed_cost,
                            storm_probability,
                            incest_penalty),
    max_episode_steps=31,
)