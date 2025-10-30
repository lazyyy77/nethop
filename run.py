#!/usr/bin/env python3
"""
Standalone runner for the Social Network Simulation System
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
import os, random
SEED = int(os.environ.get("SIM_SEED", "42"))
random.seed(SEED)
import random
import json
import networkx as nx
from Graph import Graph
from Agent import Agent
from environment import SocialEnvironment
from simulation import SimulationEngine
import asyncio
# PFEngine is optional for advanced scheduling; not required by default runner


def main():
    """Run a sample simulation"""
    print("=== Social Network Simulation System ===")

    # --- Graph Generation ---
    total_agents = 100
    max_message_batch = 20
    G = Graph(m=max_message_batch, n=total_agents // max_message_batch)
    env = SocialEnvironment(graph=G.graph)

    # Load agent profiles from agent_file.json
    with open('agent_file.json', 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    if len(profiles) < total_agents:
        print(f"Warning: agent_file.json contains only {len(profiles)} profiles but total_agents={total_agents}. Reusing profiles or truncating.")

    created_agents = []
    for i, node_id in enumerate(list(G.graph.nodes())[:total_agents]):
        if i < len(profiles):
            profile = profiles[i]
        else:
            profile = {}
            print(f"Using empty profile for agent {i} due to insufficient profiles in agent_file.json.")
        agent_id = str(node_id)
        agent = Agent(agent_id, profile)
        env.agents[agent_id] = agent
        created_agents.append(agent)

    # Prefill (warm) all agents by running their test_agent concurrently before simulation
    async def _prefill_all(agents):
        tasks = [ag.test_agent() for ag in agents]
        await asyncio.gather(*tasks)

    print("Prefilling agents (warming LLMs / prompts)...")
    asyncio.run(_prefill_all(created_agents))

    print(f"Network setup complete: {env.get_network_info()}")

    # --- Simulation ---

    print("\n--- Starting Simulation ---")
    engine = SimulationEngine(env, max_message_batch)  # Increased steps for cross-community propagation
    # run_simulation is async, run it in the event loop
    asyncio.run(engine.run_simulation())

    print("\n--- Simulation Complete ---")
    # print(f"Final stats: {engine.get_simulation_stats()}")

if __name__ == "__main__":
    main()