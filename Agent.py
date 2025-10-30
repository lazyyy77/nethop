import random
import json
from typing import List, Dict, Any, Optional, Tuple
from message import Message
from constants import EMOTION_STATES, ATTITUDE_STATES, TOPIC_TYPES
import asyncio
from event import Event
from llm import OpenAILLM

class Agent:
    """
    Agent represents an actor in the simulation.

    The constructor accepts an agent_id (string) and an optional profile dict
    (loaded from `agent_file.json`). All profile keys are attached as
    attributes for easy access (e.g. .name, .age, .persona ...).
    """
    def __init__(self, agent_id: str, profile: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        # default fields
        self.name = None
        self.age = None
        self.gender = None
        self.race = None
        self.education = None
        self.occupation = None
        self.marriage_status = None
        self.persona = None
        self.background_story = None
        self.topics = None
        
        self.llm = OpenAILLM()

        # internal simulation state
        self.emotion = random.choice(list(EMOTION_STATES)) if 'EMOTION_STATES' in globals() else None
        self.attitude = random.choice(list(ATTITUDE_STATES)) if 'ATTITUDE_STATES' in globals() else None

        if profile:
            # attach profile fields as attributes if present
            for k, v in profile.items():
                # avoid overwriting internal attributes unintentionally
                setattr(self, k, v)
        
        self.prompt = self._build_persona_prefix()

    def _build_persona_prefix(self) -> str:
        chunks = [
            f"Name: {self.name or self.agent_id}",
            f"Age: {self.age}",
            f"Gender: {self.gender}",
            f"Race: {self.race}",
            f"Education: {self.education}",
            f"Occupation: {self.occupation}",
            f"Marriage: {self.marriage_status}",
            f"Persona: {self.persona}",
            f"Background: {self.background_story}",
            f"Current emotion: {self.emotion}",
            f"Current attitude: {self.attitude}",
            f"Interested topics: {self.topics}",
        ]
        chunks = [c for c in chunks if "None" not in c]
        return "You are acting as the following agent:\n" + "\n".join(chunks) + "\n\n"

    def printAgent(self):
        print(f"Agent ID: {self.agent_id}")
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Prefix: {self.prompt}")

    def act(self) -> Optional[Dict[str, Any]]:
        """
        Produce a simple behavior dictionary. This is a placeholder; you can
        expand it to use persona, emotion, topics, etc.
        """
        behaviors = ['like', 'forward', 'comment', 'generate_new_content']
        behavior = random.choice(behaviors)

        if behavior == 'like':
            return {'type': 'like', 'message_id': None, 'agent_id': self.agent_id}
        elif behavior == 'forward':
            return {'type': 'forward', 'message_id': None, 'agent_id': self.agent_id}
        elif behavior == 'comment':
            content = f"Comment by {getattr(self, 'persona', self.name or self.agent_id)}"
            return {'type': 'comment', 'content': content, 'agent_id': self.agent_id}
        elif behavior == 'generate_new_content':
            topic = random.choice(TOPIC_TYPES) if 'TOPIC_TYPES' in globals() else 'general'
            return {'type': 'post', 'content': f"New post about {topic} by {getattr(self, 'persona', self.name or self.agent_id)}", 'topic': topic, 'agent_id': self.agent_id}

    async def run(self, event: Event, shared_balance, llm) -> Tuple[bool, bool]:


        prefix = self.prompt

        prompt1 = (
            prefix
            + f"Here is a social-media event:\n{event}\n"
            + "Do you want to spread (forward/like/retweet) this event? "
            + "Answer only 'yes' or 'no'."
        )
        forward = (await self.llm.acomplete(prompt1)).lower() == "yes"
        if not forward:
            return False, False

        if not await shared_balance.try_consume(1):
            return True, False

        prompt2 = (
            prefix
            + f"Here is a social-media event:\n{event}\n"
            + "What's your attitude towards this event now?"
        )
        
        attitude = await self.llm.acomplete(prompt2)
        
        self.topics[event.topic_type] = attitude
        
        return True, True
                

    def __repr__(self):
        return f"Agent(id={self.agent_id}, name={getattr(self, 'name', None)}, persona={getattr(self, 'persona', None)})"
