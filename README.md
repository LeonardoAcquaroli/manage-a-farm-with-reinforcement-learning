# Manage a Farm with Reinforcement Learning 👨🏻‍🌾

This repo contains the project for the course Reinforcement learning at Unimi (A.Y. 23/24).

Course Instructors: Prof. Nicolò Cesa-Bianchi and Prof. Alfio Ferrara

Course Assistants: Elisabetta Rocchetti (PhD student)

Data Science for Economics Master Degree, Università degli Studi di Milano

## Installation

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/LeonardoAcquaroli/manage-a-farm-with-rl.git`
2. Install the required dependencies: `poetry install`

## Usage

Run the `train.ipynb` notebook to train two different Reinforcement learning algorithms usign the Value Function Approximation approach (SARSA and Montecarlo) and the REINFORCE method with Advantage function (with linear functions or neural networks).
Run the `inference.ipynb` to generate an episode with the best agent that is powered by the REINFORCE method with Advantage function (Neural networks).

In particular the simple architectures for the Policy and the Value functions are depicted below.
<p float="left">
  <img src="https://github.com/LeonardoAcquaroli/manage-a-farm-with-reinforcement-learning/blob/main/media/policy_network.svg" width="45%" alt="Policy Function Architecture" />
  <img src="https://github.com/LeonardoAcquaroli/manage-a-farm-with-reinforcement-learning/blob/main/media/value_network.svg" width="45%" alt="Value Function Architecture" />
</p>


# Text of the problem
### With my edits (in bold)
You are the manager of a farm. You have an initial budget of 2000 €. Each year you have to take some decisions about how to invest you money, but you can do only one of the following things:
1. Buy one sheep: a sheep costs 1000 €
2. Growing wheat: when you choose this action, you spend 20 €
3. Do not invest: you do not spend anything (available only if the budget does not allow to grow wheat).

At the end of the year, you harvest the wheat and you sell your wool. Each sheep produces 1 wool unit that is sold for 10 €. **Selling wool has a fixed cost of 9 €.** Selling the harvested wheat instead gives you 50 €.

However, during the year, there is a probability $\alpha$ that your fields are devastated by a storm. In this case, your harvest will give you 0 €.

Moreover, if you have more than one sheep, there is a probability $\beta$ that each pair of sheep generates a new sheep.
**This is not a fixed probabilty in order to avoid sheeps to have incests. It is given by the fraction of sheeps bought over the total and it is raised to power (called incest penalty) from 2 to 5**.
Your manager career ends if you run out of money or, in any case, after 30 years, when you will retire.
You want to have a long and prosperous career.

Goal
1. Find a way to leave your heirs as much expected legacy as possible, which means that you want to learn
the best investment strategy in order to maximize your total monetary reward at the time of your retire
2. Study how $\alpha$ and $\beta$ and influence the situation you have to deal with.
