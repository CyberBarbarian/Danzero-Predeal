# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:
# TODO: 迁移后统一接口

import json

from .action import Action
from .state import State


class Ai6_agent():

    def __init__(self, id):
        self.state = State(id)
        self.action = Action(id)
        self.id = id

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        message = json.loads(str(message))
        self.state.parse(message)
        if "actionList" in message:
            act_index = self.action.parse(message)
            return act_index
