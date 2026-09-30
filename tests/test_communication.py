import pytest
import numpy as np
from config import Config
from models import MessageType
from environment import Environment
from communication import CommunicationManager


class TestCommunication:
    def test_messages_can_be_dropped(self):
        config = Config()
        config.communication_loss_rate = 1.0
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        result = comm.send_message(1, 2, MessageType.POSITION_UPDATE, {"x": 10, "y": 20})
        assert result == False

    def test_local_operation_continues_during_communication_loss(self):
        config = Config()
        config.communication_loss_rate = 0.5
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        env.tick = 0
        
        messages_sent = 0
        for i in range(10):
            if comm.send_message(1, 2, MessageType.POSITION_UPDATE, {}):
                messages_sent += 1
        
        received = comm.receive_messages(2)
        messages_received = len(received)
        
        assert messages_received <= messages_sent

    def test_message_ttl_expires(self):
        config = Config()
        config.message_ttl = 1
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        
        comm.send_message(1, 2, MessageType.POSITION_UPDATE, {"x": 10, "y": 20})
        comm.update()
        
        received = comm.receive_messages(2)
        assert len(received) == 0

    def test_communication_loss_rate_affects_delivery(self):
        config = Config()
        config.communication_loss_rate = 0.9
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        
        sent_count = 0
        for i in range(100):
            if comm.send_message(1, 2, MessageType.POSITION_UPDATE, {}):
                sent_count += 1
        
        assert sent_count <= 20

    def test_communication_manager_initialization(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        
        assert comm.loss_rate == 0.0
        assert comm.message_counter == 0
        assert comm.messages == {}