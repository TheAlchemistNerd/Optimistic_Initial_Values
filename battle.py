from bandit2 import Bandit
from compare_agents import compare_agents
from epsilon_greedy_agent_2 import EpsilonGreedyAgent
from optimistic_initial_values import OptimisticInitialValuesAgent

agents = [
    EpsilonGreedyAgent(),
    OptimisticInitialValuesAgent(max_reward=20),
]
bandits = [
  Bandit(m=mu, lower_bound=0, upper_bound=10)
  for mu in [3, 5, 7, 9]
]
iterations = 70
compare_agents(agents, bandits, iterations)