import json
import random
import networkx as nx
from typing import Dict, List, Optional
from Agent import Agent
from message import Message
from event import Event
import asyncio

class SharedBalance:
    def __init__(self, total: int):
        self._remain = total
        # don't create an asyncio.Lock() here (can raise if no event loop);
        # create it lazily inside the first async call where the loop exists
        self._lock = None
    
    def reset(self, total: int):
        self._remain = total

    async def try_consume(self, n: int = 1) -> bool:
        # lazily create the lock in coroutine context where an event loop exists
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._remain >= n:
                self._remain -= n
                return True
            return False

    @property
    def remain(self) -> int:
        return self._remain

class SocialEnvironment:
    def __init__(self, graph: nx.DiGraph = None):
        self.Graph = graph if graph is not None else nx.DiGraph()
        self.graph = graph.graph if graph.graph is not None else nx.DiGraph()
        self.agents: Dict[str, Agent] = {}
        self.message_pool: List[Message] = []
        self.event_pool: List[Event] = []
        self.load_events_from_file()

    def  add_agent(self, agent: Agent):
        self.agents[agent.agent_id] = agent
        if not self.graph.has_node(agent.agent_id):
            self.graph.add_node(agent.agent_id, agent=agent, community_id=None)
        else:
            # Ensure agent object is attached to the node if it was pre-created
            self.graph.nodes[agent.agent_id]['agent'] = agent

    def add_following(self, follower_id: str, followee_id: str):
        if follower_id in self.agents and followee_id in self.agents:
            self.graph.add_edge(follower_id, followee_id)

    def generate_event(self, topic_type: str, content: str) -> Event:
        event = Event(topic_type, content)
        self.event_pool.append(event)
        return event

    def load_events_from_file(self, filepath: str = './event_file.json', clear_existing: bool = False):
        """Load events from a JSON file into the environment's event_pool.

        The JSON file is expected to be a mapping from topic category (string)
        to a list of event text strings. For each text an Event instance will
        be created and appended to `self.event_pool`.

        If `clear_existing` is True, the existing `event_pool` will be cleared
        before loading.
        """
        if clear_existing:
            self.event_pool = []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # data is expected to be { category: [text, ...], ... }
        for category, texts in data.items():
            if not isinstance(texts, list):
                continue
            for text in texts:
                try:
                    evt = Event(category, text)
                except ValueError:
                    # Skip events whose category is not in allowed TOPIC_TYPES
                    continue
                self.event_pool.append(evt)

    def output_event(self, category: Optional[str] = None) -> Optional[Event]:
        """Return a random Event.

        If `category` is provided, pick a random event from that category (if any).
        Otherwise, pick a random category present in `event_pool`, then a random
        event from that category.

        Returns None if no suitable event is available.
        """
        if not self.event_pool:
            return None

        if category:
            candidates = [e for e in self.event_pool if e.topic_type == category]
            if not candidates:
                return None
            return random.choice(candidates)

        # pick a random category among loaded events
        categories: Dict[str, List[Event]] = {}
        for e in self.event_pool:
            categories.setdefault(e.topic_type, []).append(e)

        if not categories:
            return None

        chosen_cat = random.choice(list(categories.keys()))
        return random.choice(categories[chosen_cat])

    def distribute_message(self, message: Message):
        self.message_pool.append(message)

    def get_message_feed(self) -> List[Message]:
        return self.message_pool

    def get_network_info(self):
        """Get basic network statistics"""
        return {
            'num_agents': len(self.agents),
            'num_edges': self.graph.number_of_edges(),
            'num_messages': len(self.message_pool),
            'num_events': len(self.event_pool)
        }

    def __repr__(self):
        info = self.get_network_info()
        return f"SocialEnvironment(agents={info['num_agents']}, edges={info['num_edges']})"