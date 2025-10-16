import numpy as np
import json


class RandomAgent():
    def __init__(self, id=0):
        self.id = id

    def received_message(self, message):
        try:
            message = json.loads(str(message))
        except (json.JSONDecodeError, TypeError):
            return 0
            
        if "actionList" in message and len(message['actionList']) > 0:
            # Ensure we have valid legal actions
            index_range = message.get('indexRange', 0)
            action_list_len = len(message['actionList'])
            
            # Use the minimum of indexRange+1 and actual action list length
            upper = min(max(1, index_range + 1), action_list_len)
            lower = 0
            
            # Double check that we have actions and upper > lower
            if upper > lower and action_list_len > 0:
                try:
                    return np.random.randint(lower, upper)
                except ValueError as e:
                    print(f"RandomAgent ValueError: {e}")
                    print(f"RandomAgent debug: upper={upper}, lower={lower}, indexRange={index_range}, actionList_len={action_list_len}")
                    print(f"Message keys: {list(message.keys())}")
                    return 0
            else:
                return 0  # Default to first action (usually PASS)
        else:
            return 0  # No legal actions available, return first action