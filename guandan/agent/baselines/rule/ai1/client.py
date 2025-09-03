# TODO: 重构为统一BaseAgent接口，整合特征/动作空间
# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:

import json
from agent.baselines.rule.ai1.state import State
from agent.baselines.rule.ai1.action import Action


class Ai1_agent():

    def __init__(self, id=0):

        self.state = State("client" + str(id))
        self.action = Action("client" + str(id))
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
            return act_index
