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
        self.recv_time = 0
        
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

    async def run(self, event: Event, shared_balance) -> Tuple[bool, bool]:


        prefix = self.prompt

        prompt1 = (
            prefix
            + f"Here is a social-media event:\n{event}\n"
            + "Do you want to spread (forward/like/retweet) this event? "
            + "Answer only 'yes' or 'no'."
        )
        forward = (await self.llm.acomplete(prompt1, agent_id=str(self.agent_id))).lower() == "yes"
        print(f"Agent {self.agent_id} forward decision: {forward}")
        if not forward:
            return False, False

        prompt2 = (
            prefix
            + f"Here is a social-media event:\n{event}\n"
            + "What's your attitude towards this event now?"
        )

        attitude = await self.llm.acomplete(prompt2, agent_id=str(self.agent_id))
        print(f"Agent {self.agent_id} attitude")
        
        # self.topics[event.topic_type] = attitude
        
        if shared_balance is not None:
            if not await shared_balance.try_consume(1):
                return True, False
        
        return True, True
                

    def __repr__(self):
        return f"Agent(id={self.agent_id}, name={getattr(self, 'name', None)}, persona={getattr(self, 'persona', None)})"

    async def test_agent(self):
        event = Event(content="Test event for agent warming.", topic_type="Technology Privacy")
        prompt1 = (
            self.prompt
            + f"Here is a social-media event:\n{event}\n"
            + "Do you want to spread (forward/like/retweet) this event? "
            + "Answer only 'yes' or 'no'."
        )
        self.llm.acomplete(prompt1, agent_id=str(self.agent_id))

if __name__ == "__main__":
    print("Testing Agent...")
    profile = {
        "name": "Alice",
        "age": 30,
        "gender": "Female",
        "persona": "A friendly and curious individual who loves to explore new ideas.",
        "background_story": "Alice grew up in a small town and moved to the city to pursue her dreams.",
        "topics": {
            "Gender Discrimination": "Firmly believes in treating everyone with respect and fairness, regardless of gender.",
            "Climate Change": "Deeply worried about the environmental impact and its effect on communities worldwide.",
            "Racial Discrimination": "Strongly opposes any form of discrimination and promotes unity among people of all races.",
            "Technology Privacy": "Concerned about the misuse of personal information and advocates for strong privacy protections.",
            "Economic Inequality": "Supports fair distribution of resources and believes in helping those less fortunate."
        }
    }
    agent = Agent(agent_id="1", profile=profile)
    agent.printAgent()
    agent.llm = OpenAILLM()
    event = Event(
        content="New breakthrough in AI technology announced!",
        topic_type="Technology Privacy"
    )
    async def test():
        await agent.run(event, shared_balance=None, llm=agent.llm)

    asyncio.run(test())
    
    print("Agent test completed.")