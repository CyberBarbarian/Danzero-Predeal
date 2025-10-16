# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:
# TODO: Unify interface after migration

import json

from .action import Action
from .state import State
from ....logging_config import get_agent_logger


class Ai6_agent():

    def __init__(self, id):
        self.state = State(id)
        self.action = Action(id)
        self.id = id
        self.logger = get_agent_logger("ai6")

    def opened(self):
        pass

    def closed(self, code, reason=None):
        self.logger.info(f"Agent {self.id} closed down: code={code}, reason={reason}")

    def received_message(self, message):
        try:
            message = json.loads(str(message))
            self.state.parse(message)
            if "actionList" in message and len(message["actionList"]) > 0:
                act_index = self.action.parse(message)
                return act_index
            else:
                return 0  # No legal actions available
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            self.logger.error(f"Error in ai6 agent {self.id}: {e}")
            return 0
