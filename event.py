from datetime import datetime
from constants import TOPIC_TYPES


class Event:
    def __init__(self, topic_type: str, content: str, timestamp: datetime = None,
                 authenticity: float = 1.0, relevance: float = 1.0, jump: bool = False):
        if topic_type not in TOPIC_TYPES:
            raise ValueError(f"Invalid topic_type. Must be one of {TOPIC_TYPES}")
        self.topic_type = topic_type
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.authenticity = authenticity
        self.relevance = relevance
        self.jump = jump

    def printEvent(self):
        print(f"Event(topic_type={self.topic_type}, content={self.content}, "
              f"timestamp={self.timestamp}, authenticity={self.authenticity}, "
              f"relevance={self.relevance}, jump={self.jump})")

    def __repr__(self):
        return f"Event(topic={self.topic_type}, content={self.content[:50]}...)"