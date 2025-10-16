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
from ....logging_config import get_agent_logger


class Ai3_agent():

    def __init__(self, id):
        self.state = State()
        self.action = Action()
        self.id = id
        self.logger = get_agent_logger("ai3")

    def opened(self):
        pass

    def closed(self, code, reason=None):
        self.logger.info(f"Agent {self.id} closed down: code={code}, reason={reason}")

    def received_message(self, message):
        try:
            message = json.loads(str(message))                                    # 先序列化收到的消息，转为Python中的字典
            # print(self.id, message)
            self.state.parse(message)                                             # 调用状态对象来解析状态
            if 'myPos' in message:
                global pos
                pos = message['myPos']
            if "actionList" in message and len(message["actionList"]) > 0:    # 需要做出动作选择时调用动作对象进行解析
                #由AI进行选择，座位号随时读取
                act_index = self.action.parse_AI(message, pos)
                return act_index
            else:
                return 0  # No legal actions available
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            self.logger.error(f"Error in ai3 agent {self.id}: {e}")
            return 0
