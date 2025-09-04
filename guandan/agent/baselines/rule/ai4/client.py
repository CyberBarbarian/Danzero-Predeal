# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:
# TODO: Unify interface after migration


import json
from .state import State
from .action import Action


class Ai4_agent():
    def __init__(self, id=0):
        self.state = State("client1")
        self.action = Action("client1")
        self.id = id

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        message = json.loads(str(message))
        self.state.parse(message)
        if "actionList" in message:
            act_index = self.action.rule_parse(message, self.state._myPos, self.state.remain_cards, self.state.history,
                                               self.state.remain_cards_classbynum, self.state.pass_num,
                                               self.state.my_pass_num, self.state.tribute_result)
            print(act_index)
            return act_index
