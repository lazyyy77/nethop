from datetime import datetime
import uuid
import random
from typing import Dict, Any
from event import Event
from constants import MESSAGE_TYPES


class Message:
    def __init__(self, sender_id: str, content: str, message_type: str,
                 timestamp: datetime = None, event: Event = None):
        if message_type not in MESSAGE_TYPES:
            raise ValueError(f"Invalid message_type. Must be one of {MESSAGE_TYPES}")
        self.message_id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.content = content
        self.type = message_type
        self.timestamp = timestamp or datetime.now()
        self.propagation_count = 0
        self.importance = 0.0
        self.event = event

    def increment_propagation(self):
        self.propagation_count += 1

    def calculate_importance(self, user_attributes: Dict[str, Any]):
        # Simplified importance calculation based on cosine similarity stub
        # In real implementation, would use actual similarity metrics
        base_importance = self.propagation_count * self.event.relevance if self.event else self.propagation_count
        self.importance = base_importance * random.uniform(0.5, 1.5)  # Random factor for simulation

    def __repr__(self):
        return f"Message(id={self.message_id}, sender={self.sender_id}, type={self.type}, content={self.content[:50]}...)"