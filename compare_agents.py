from typing import List
from agent2 import Agent
from bandit2 import Bandit
import numpy as np
import matplotlib.pyplot as plt
import logging as logger
from optimistic_initial_values import OptimisticInitialValuesAgent


def compare_agents(agents: List[Agent], bandits: List[Bandit], iterations: int, show_plot=True):
    for agent in agents:
        logger.info("Running for agent = %s", agent)
        agent.bandits = bandits
        agent.take_actions(iterations)
        if show_plot:
            plt.plot(np.cumsum(agent.rewards_log.all_rewards), label=str(agent))
    if show_plot:
        plt.xlabel("iteration")
        plt.ylabel("total rewards")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

def main():
    bandits = [
        Bandit(m = mu, lower_bound=0, upper_bound=10)
        for mu in [3, 5, 7, 9]
    ]

    agents = [
        OptimisticInitialValuesAgent(max_reward=r)
        for r in [15, 20, 50, 100, 1000]
    ]

    iterations = 70
    compare_agents(agents, bandits, iterations)

if __name__ == '__main__':
    main()