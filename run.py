#!/usr/bin/env python3
"""
Standalone runner for the Social Network Simulation System
"""

import random
import json
import networkx as nx
from generator import generate_lfr_graph
from Graph import Graph
from Agent import Agent
from environment import SocialEnvironment
from simulation import SimulationEngine


def main():
    """Run a sample simulation"""
    print("=== Social Network Simulation System ===")

    # --- Graph Generation ---
    total_agents = 10
    G = Graph(vertex=total_agents)
    env = SocialEnvironment(graph=G.graph)

    # Load agent profiles from agent_file.json
    with open('agent_file.json', 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    if len(profiles) < total_agents:
        print(f"Warning: agent_file.json contains only {len(profiles)} profiles but total_agents={total_agents}. Reusing profiles or truncating.")

    for i, node_id in enumerate(list(G.graph.nodes())[:total_agents]):
        if i < len(profiles):
            profile = profiles[i]
        else:
            profile = {}
            print(f"Using empty profile for agent {i} due to insufficient profiles in agent_file.json.")
        agent_id = str(node_id)
        agent = Agent(agent_id, profile)
        env.agents[agent_id] = agent

    print(f"Network setup complete: {env.get_network_info()}")

    # --- Simulation ---
    max_message_batch = 10
    print("\n--- Starting Simulation ---")
    engine = SimulationEngine(env, max_message_batch)  # Increased steps for cross-community propagation
    engine.run_simulation()

    print("\n--- Simulation Complete ---")
    # print(f"Final stats: {engine.get_simulation_stats()}")

if __name__ == "__main__":
    main()