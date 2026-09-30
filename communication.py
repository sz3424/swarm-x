from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from config import Config
from models import Agent, Message, MessageType
from environment import Environment


class CommunicationManager:
    def __init__(self, config: Config, env: Environment):
        self.config = config
        self.env = env
        self.messages: Dict[int, List[Message]] = {}
        self.message_counter = 0
        self.loss_rate = config.communication_loss_rate

    def send_message(self, sender_id: int, receiver_id: int, message_type: MessageType, payload: Dict) -> bool:
        if np.random.random() > self.loss_rate:
            message = Message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                timestamp=self.env.tick if hasattr(self.env, 'tick') else 0,
                message_type=message_type,
                sequence_number=self.message_counter,
                payload=payload,
                ttl=self.config.message_ttl
            )
            self.message_counter += 1

            if receiver_id not in self.messages:
                self.messages[receiver_id] = []
            self.messages[receiver_id].append(message)
            return True
        return False

    def receive_messages(self, agent_id: int) -> List[Message]:
        if agent_id not in self.messages:
            return []

        received = []
        for msg in self.messages[agent_id]:
            if msg.ttl > 0:
                received.append(msg)

        self.messages[agent_id] = []
        return received

    def update(self) -> None:
        for agent_id in list(self.messages.keys()):
            for msg in self.messages[agent_id]:
                msg.ttl -= 1
            self.messages[agent_id] = [m for m in self.messages[agent_id] if m.ttl > 0]

    def set_loss_rate(self, rate: float) -> None:
        self.loss_rate = rate
