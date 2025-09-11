# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:
# 对抗AI1号

from random import randint

import json

pos = 0

from .action import Action
from .state import State


class Ai3_agent():

    def __init__(self, id):
        self.state = State()
        self.action = Action()
        self.id = id

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        message = json.loads(str(message))                                    # 先序列化收到的消息，转为Python中的字典
        # print(self.id, message)
        self.state.parse(message)                                             # 调用状态对象来解析状态
        if 'myPos' in message:
            global pos
            pos = message['myPos']
        if "actionList" in message:    # 需要做出动作选择时调用动作对象进行解析
            #由AI进行选择，座位号随时读取
            act_index = self.action.parse_AI(message, pos)
            # try:
            #     act_index = self.action.parse_AI(message,pos)
            # except:
            #     act_index = randint(0, message['indexRange'])
            return act_index
