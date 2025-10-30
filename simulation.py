import random
from environment import SocialEnvironment, SharedBalance
from message import Message
from constants import TOPIC_TYPES
import asyncio
import networkx as nx
import time

class SimulationEngine:
    def __init__(self, environment: SocialEnvironment, max_message_batch: int = 10, message_source: int = 5):
        self.environment = environment
        self.max_message_batch = max_message_batch
        self.message_source = message_source
        self.shared_balance = SharedBalance(total=self.max_message_batch - len(self.environment.agents))
        # track how many times each node (by int id) has been visited/activated
        self.visit_counts = {}
        print(f"Initialized SimulationEngine with max_message_batch={self.max_message_batch}, message_source={self.message_source}, SharedBalance={self.shared_balance.remain}")

    async def run_simulation(self):
        step = 0
        active_agent = []
        # record total simulation start time
        t_start = time.perf_counter()
        while True:
            print(f"-------- Step {step} --------")
            # print(self.environment.output_event())
            # self.environment.agents[str(step)].printAgent()
            if step == 0:
                g = self.environment.graph
                selected_vertex = g.get_chain_heads()
                print(f"Selected vertices for posting messages: {selected_vertex}")
                first_ts_agents = []
                event = self.environment.output_event()
                for node_id in selected_vertex:
                    agent = self.environment.agents.get(str(node_id))
                    if agent:
                        first_ts_agents.append(agent)
                results = await asyncio.gather(*(ag.run(event, self.shared_balance) for ag in first_ts_agents))
                for i in first_ts_agents:
                    if results[first_ts_agents.index(i)][0]:
                        id = int(i.agent_id)
                        dst = self.shortest_out_edge_id(self.environment.graph, id, 0)
                        if dst is not None:
                            agent_obj = self.environment.agents.get(str(dst))
                            if agent_obj is not None and agent_obj not in active_agent:
                                active_agent.append(agent_obj)
                            # increment visit count for the selected node
                            try:
                                did = int(dst)
                            except Exception:
                                did = dst
                            self.visit_counts[did] = self.visit_counts.get(did, 0) + 1
                    if results[first_ts_agents.index(i)][1] or self.shared_balance.remain > 0:
                        await self.shared_balance.try_consume(1)
                        id = int(i.agent_id)
                        dst = self.shortest_out_edge_id(self.environment.graph, id, 1)
                        if dst is not None:
                            agent_obj = self.environment.agents.get(str(dst))
                            if agent_obj is not None and agent_obj not in active_agent:
                                active_agent.append(agent_obj)
                            try:
                                did = int(dst)
                            except Exception:
                                did = dst
                            self.visit_counts[did] = self.visit_counts.get(did, 0) + 1
            else:
                if not active_agent:
                    print("No more active agents to process. Ending simulation.")
                    break
                current_agents = active_agent.copy()
                active_agent = []
                results = await asyncio.gather(*(ag.run(event, self.shared_balance) for ag in current_agents if ag is not None))
                for i in current_agents:
                    if i is None:
                        continue
                    if results[current_agents.index(i)][0]:
                        id = int(i.agent_id)
                        dst = self.shortest_out_edge_id(self.environment.graph, id, 0)
                        if dst is not None:
                            agent_obj = self.environment.agents.get(str(dst))
                            if agent_obj is not None and agent_obj not in active_agent:
                                active_agent.append(agent_obj)
                            try:
                                did = int(dst)
                            except Exception:
                                did = dst
                            self.visit_counts[did] = self.visit_counts.get(did, 0) + 1
                    if results[current_agents.index(i)][1] or self.shared_balance.remain > 0:
                        await self.shared_balance.try_consume(1)
                        id = int(i.agent_id)
                        dst = self.shortest_out_edge_id(self.environment.graph, id, 1)
                        if dst is not None:
                            agent_obj = self.environment.agents.get(str(dst))
                            if agent_obj is not None and agent_obj not in active_agent:
                                active_agent.append(agent_obj)
                            try:
                                did = int(dst)
                            except Exception:
                                did = dst
                            self.visit_counts[did] = self.visit_counts.get(did, 0) + 1
            step += 1
            print(len(self.visit_counts))
            if step >= 20:
                break

        all_time = time.perf_counter() - t_start
        all_num = sum(self.visit_counts.values())
        print(f"[Node {all_num}][Time {all_time:.4f}][Avg {all_time / all_num if all_num > 0 else 0:.6f}]")

    def shortest_out_edge_id(self, g: nx.DiGraph, node_id, order):
        """
        Return the successor of `node_id` ordered by edge weight at index `order`,
        but skip any successor whose corresponding agent/node has been visited
        two times or more (according to self.visit_counts). If not enough
        eligible successors exist, return None.
        """
        if node_id not in g or g.out_degree(node_id) == 0:
            return None

        # sort successors by edge weight (ascending)
        successors = sorted(g.successors(node_id), key=lambda v: g[node_id][v].get('weight', 0))

        # filter out successors that have been visited >= 2 times
        eligible = []
        for v in successors:
            # node ids in graph might be ints; use int for visit_counts keys
            try:
                vid = int(v)
            except Exception:
                vid = v
            if self.visit_counts.get(vid, 0) >= 2:
                # skip this successor
                continue
            eligible.append(v)

        if order < 0 or order >= len(eligible):
            return None

        return eligible[order]


    def postMessage(self, source: int = 10):
        g = self.environment.graph
        selected_vertex = [n for n, d in sorted(g.out_degree(), key=lambda t: t[1], reverse=True)[:source]]
        print(f"Selected vertices for posting messages: {selected_vertex}")
        first_ts_agents = []
        event_list = []
        for node_id in selected_vertex:
            agent = self.environment.agents.get(str(node_id))
            if agent:
                first_ts_agents.append(agent)
                event = self.environment.output_event()
                if event:
                    event_list.append(event)
        
    
    def simulate_timestep(self):
        pass
    
    def simulate_first_timestep(self):
        pass
                    
    def get_simulation_stats(self):
        """Get statistics about the simulation"""
        return {
            'max_message_batch': self.max_message_batch,
            'network_info': self.environment.get_network_info()
        }