# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:
# TODO: Unify interface after migration


import json
from .state import State
from .action import Action
from ....logging_config import get_agent_logger


class Ai4_agent():
    def __init__(self, id=0):
        self.state = State("client1")
        self.action = Action("client1")
        self.id = id
        self.logger = get_agent_logger("ai4")

    def opened(self):
        pass

    def closed(self, code, reason=None):
        self.logger.info(f"Agent {self.id} closed down: code={code}, reason={reason}")

    def received_message(self, message):
        try:
            message = json.loads(str(message))
            self.state.parse(message)
            if "actionList" in message and len(message["actionList"]) > 0:
                act_index = self.action.rule_parse(message, self.state._myPos, self.state.remain_cards, self.state.history,
                                                   self.state.remain_cards_classbynum, self.state.pass_num,
                                                   self.state.my_pass_num, self.state.tribute_result)
                self.logger.debug(f"Agent {self.id} selected action index: {act_index}")
                return act_index
            else:
                return 0  # No legal actions available
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            self.logger.error(f"Error in ai4 agent {self.id}: {e}")
            return 0
