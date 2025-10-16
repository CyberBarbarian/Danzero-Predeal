# TODO: Interface to be unified after migration
# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu
# @File       : client.py
# @Description:
import json
from .action import Action
from .state import State
from ....logging_config import get_agent_logger


class Ai2_agent():
    def __init__(self, id=0):
        self.state = State("client" + str(id))
        self.action = Action("client" + str(id))
        self.id = id
        self.logger = get_agent_logger("ai2")

    def opened(self):
        pass

    def closed(self, code, reason=None):
        self.logger.info(f"Agent {self.id} closed down: code={code}, reason={reason}")

    def received_message(self, message):
        try:
            message = json.loads(str(message))                                    # 先序列化收到的消息，转为Python中的字典
            # print(self.id, message)
            self.state.parse(message)                                             # 调用状态对象来解析状态
            if "actionList" in message and len(message["actionList"]) > 0:                                          # 需要做出动作选择时调用动作对象进行解析
                if message["stage"] == "play":
                    act_index = self.action.GetIndexFromPlay(message, self.state.retValue)
                elif message["stage"] == "back":
                    act_index = self.action.GetIndexFromBack(message, self.state.retValue)
                else:
                    act_index = self.action.parse(message)
                return act_index
            else:
                return 0  # No legal actions available
        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            self.logger.error(f"Error in ai2 agent {self.id}: {e}")
            return 0
