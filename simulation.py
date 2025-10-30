import random
from environment import SocialEnvironment, SharedBalance
from message import Message
from constants import TOPIC_TYPES
import asyncio
import networkx as nx

class SimulationEngine:
    def __init__(self, environment: SocialEnvironment, max_message_batch: int = 10, message_source: int = 10):
        self.environment = environment
        self.max_message_batch = max_message_batch
        self.message_source = message_source
        self.shared_balance = SharedBalance(total=self.max_message_batch - len(self.environment.agents))
        print(f"Initialized SimulationEngine with max_message_batch={self.max_message_batch}, message_source={self.message_source}, SharedBalance={self.shared_balance.remain}")

    async def run_simulation(self):
        step = 0
        active_agent = []
        while True:
            print(f"-------- Step {step} --------")
            # print(self.environment.output_event())
            # self.environment.agents[str(step)].printAgent()
            if step == 0:
                g = self.environment.graph()
                selected_vertex = [n for n, d in sorted(g.out_degree(), key=lambda t: t[1], reverse=True)[:self.message_source]]
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
                        dst = self.shortest_out_edge_id(self.environment.graph(), id, 0)
                        if dst is not None:
                            active_agent.append(self.environment.agents.get(str(dst)))
                    if results[first_ts_agents.index(i)][1] or self.shared_balance.remain > 0:
                        self.shared_balance.try_consume(1)
                        id = int(i.agent_id)
                        dst = self.shortest_out_edge_id(self.environment.graph(), id, 1)
                        if dst is not None:
                            active_agent.append(self.environment.agents.get(str(dst)))
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
                        dst = self.shortest_out_edge_id(self.environment.graph(), id, 0)
                        if dst is not None:
                            active_agent.append(self.environment.agents.get(str(dst)))
                    if results[current_agents.index(i)][1] or self.shared_balance.remain > 0:
                        self.shared_balance.try_consume(1)
                        id = int(i.agent_id)
                        dst = self.shortest_out_edge_id(self.environment.graph(), id, 1)
                        if dst is not None:
                            active_agent.append(self.environment.agents.get(str(dst)))
            
            step += 1
            if step >= 20:
                break
            # # Generate some events randomly
            # if random.random() > 0.5:
            #     topic = random.choice(TOPIC_TYPES)
            #     event = self.environment.generate_event(topic, f"Event about {topic}")
            #     # Send to random active agent
            #     if self.environment.agents:
            #         active_agent_id = random.choice(list(self.environment.agents.keys()))
            #         message = Message(active_agent_id, event.content, 'post', event=event)
            #         self.environment.distribute_message(message)

            # message_feed = self.environment.get_message_feed()

            # for agent_id, agent in self.environment.agents.items():
            #     relevant_messages = agent.perceive(message_feed)
            #     for msg in relevant_messages:
            #         agent.update_state(msg)

            #     behavior = agent.act()
            #     if behavior:
            #         if behavior['type'] == 'post':
            #             new_message = Message(agent_id, behavior['content'], 'post')
            #             self.environment.distribute_message(new_message)
            #         elif behavior['type'] == 'comment':
            #             new_message = Message(agent_id, behavior['content'], 'comment')
            #             self.environment.distribute_message(new_message)
            #         # Handle like and forward similarly (could be extended)

            # Clear message pool for next step (simplified)
            # self.environment.message_pool = []

    def shortest_out_edge_id(g: nx.DiGraph, node_id, order):
        if node_id not in g or g.out_degree(node_id) == 0:
            return None
        return sorted(g.successors(node_id),
                    key=lambda v: g[node_id][v]['weight'])[order]


    def postMessage(self, source: int = 10):
        g = self.environment.graph()
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