"""
Content: Engine for Guandan.
Author : Lu Yudong
"""

import numpy as np
import time
from copy import deepcopy
from .player import Player
from .card_deck import CardDeck
from .context import Context
from .table import Table
from .utils import legalaction, vector2card, RANK2, tribute_legal, anti_tribute, back_legal, CardToNum, give_type


class GameEnv(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.ctx.table = Table()
        self.ctx.card_decks = CardDeck()
        self.ctx.players = {}
        self.ctx.players_id_list = []
        for i in range(4):
            self.ctx.players[i] = Player(i)
            self.ctx.players_id_list.append(i)
        self.ctx.wind = False   # 接风判断
        self.ctx.win_order = []
        self.wind_num = 0
        self.round_end = False
        self.episode_end = False
        self.tribute_res = []
        self.back_res = []
        self.anti_tri = False

    def deal_cards(self):
        self.ctx.table.join(deepcopy(self.ctx.players_id_list))
        self.ctx.card_decks.init_deck(2)  # 2副牌
        deal_card_lists = self.ctx.card_decks.deal(len(self.ctx.players_id_list))
        for i in range(4):
            self.ctx.players[i].update_cards(deal_card_lists[i])
            self.ctx.players[i].update_rank(self.ctx.cur_rank)  # my_rank用数字，cur_rank用字符
            self.ctx.players[i].have_cards = True

    def battle_init(self):
        self.deal_cards()  # Restore all players to table and deal new cards
        if len(self.ctx.win_order) == 0:
            self.ctx.player_waiting = np.random.randint(0, 4)
        else:
            self.tribute_phase()
        print('round end, win order', self.ctx.win_order, self.ctx.cur_rank, self.ctx.players[0].my_rank, self.ctx.players[1].my_rank)
        time.sleep(0.5)
        self.ctx.steps = 0
        self.ctx.wind = False
        self.wind_num = 0
        self.round_end = False
        
        # Check for episode completion BEFORE resetting win_order
        # Episode ends only when a team wins at rank A with correct conditions
        if self.ctx.cur_rank == 'A' and len(self.ctx.win_order) >= 2:
            first = self.ctx.win_order[0]  # Banker
            banker_partner = (first + 2) % 4
            
            # Check if partner is Follower (2nd place)
            if len(self.ctx.win_order) >= 2 and self.ctx.win_order[1] == banker_partner:
                self.episode_end = True
                print(f"Game won at rank A: Banker {first}, Partner {banker_partner} is Follower")
            # Check if partner is Third (3rd place)
            elif len(self.ctx.win_order) >= 3 and self.ctx.win_order[2] == banker_partner:
                self.episode_end = True
                print(f"Game won at rank A: Banker {first}, Partner {banker_partner} is Third")
        
        # Reset win order AFTER checking for episode completion
        self.ctx.win_order = []
        self.ctx.trick_pass = 0
        self.ctx.recv_wind = False
        self.tribute_res = []
        self.back_res = []
        self.ctx.last_action = None
        self.ctx.last_max_action = None
        self.ctx.last_max_playid = None
        self.ctx.last_playid = None
        self.anti_tri = False

    def step(self, action=None):
        print('waiting player', self.ctx.player_waiting)
        last_type, last_value = give_type(self.ctx)
        legalactions = legalaction(self.ctx, last_type=last_type, last_value=last_value)
        # legalactions = legalaction(self.ctx)
        if action is None:
            # default random action
            if len(legalactions) > 0:
                action = np.random.randint(0, len(legalactions))
            else:
                # No legal actions available, use empty action
                self.update([])
                print('played', [], [])
                return 0
        
        if action < len(legalactions):
            action_to_use = legalactions[action]
            self.update(action_to_use)
            print('played', vector2card(action_to_use[:54]), action_to_use)
        else:
            # Action index out of range, use first action or empty
            action_to_use = legalactions[0] if legalactions else []
            self.update(action_to_use)
            print('played', vector2card(action_to_use[:54]) if action_to_use else [], action_to_use)
        return 0

    def update(self, action): # action最后一项是last_type
        self.ctx.last_action = action
        self.ctx.last_playid = self.ctx.player_waiting
        self.ctx.recv_wind = False
        if len(action) == 0:
            self.ctx.trick_pass += 1
        else:
            self.ctx.trick_pass = 0
            self.ctx.last_max_action = self.ctx.last_action   # 只要不是新一轮，没人过，上一个动作就是当前轮的最大动作
            self.ctx.last_max_playid = self.ctx.player_waiting
        if self.ctx.wind:   # 如果当前是一个人出完牌的接风轮。有人出了牌，则接风失效
            if len(action) > 0:
                self.ctx.wind = False
                self.wind_num = 0
        self.ctx.players[self.ctx.player_waiting].play(action)
        # print(self.ctx.card_decks.sortcards(self.ctx.players[self.ctx.player_waiting].handcards_in_str))
        if not self.ctx.players[self.ctx.player_waiting].have_cards:   # 牌出完了，就把这个人移走
            self.ctx.table.detach(self.ctx.player_waiting)
            self.ctx.win_order.append(self.ctx.player_waiting)
            self.ctx.wind = True     # 有人出完牌，进入接风轮
            self.wind_num = self.ctx.table.players_on_table_numb
            # Check if round should end AFTER removing player from table
            self.whether_end()   # 判断游戏是否结束
        if not self.round_end:
            if self.ctx.wind:
                # 如果更新完动作后还是接风轮
                if self.wind_num == 0:  # 表示接风轮的最后一个人都做完动作了（因为进到这里wind是true，说明最后一个人也是过，然后就到对家接风）
                    self.ctx.wind = False
                    self.ctx.player_waiting = (self.ctx.table.players_last_end[-1] + 2) % 4
                    self.ctx.recv_wind = True
                else:
                    while (self.ctx.player_waiting + 1) % 4 not in self.ctx.table.players_on_table_id:  # 如果下家已经出完牌了，则递补继续向下
                        self.ctx.player_waiting = (self.ctx.player_waiting + 1) % 4
                    self.ctx.player_waiting = (self.ctx.player_waiting + 1) % 4
                    self.wind_num -= 1
            else:
                while (self.ctx.player_waiting + 1) % 4 not in self.ctx.table.players_on_table_id:  # 如果下家已经出完牌了，则递补继续向下
                    self.ctx.player_waiting = (self.ctx.player_waiting + 1) % 4
                self.ctx.player_waiting = (self.ctx.player_waiting + 1) % 4

    def whether_end(self):
        condition1 = len(self.ctx.table.players_on_table_id) == 1      # 3个人都出完了或者剩的俩是对家
        condition2 = len(self.ctx.table.players_on_table_id) == 2 and \
                     abs(self.ctx.table.players_on_table_id[0] - self.ctx.table.players_on_table_id[1]) == 2
        if condition1 or condition2:
            self.round_end = True
            for i in self.ctx.table.players_on_table_id:
                self.ctx.win_order.append(i)
            self.upgrade()

    def determine_winner(self):
        """
        Determine the winner based on Guandan rules.
        In Guandan, only the team with the FIRST PLACE player wins.
        The teammate's position determines the rank promotion.
        """
        if len(self.ctx.win_order) < 1:
            return  # Not enough players to determine winner
            
        first_place = self.ctx.win_order[0]
        
        # Define teams: Team 1 (players 0,2) vs Team 2 (players 1,3)
        team1_players = [0, 2]
        team2_players = [1, 3]
        
        # In Guandan, only the team with the FIRST PLACE player wins
        if first_place in team1_players:
            self.ctx.winner_team = 'team1'
            self.ctx.loser_team = 'team2'
            self.ctx.game_result = 'team1_win'
        elif first_place in team2_players:
            self.ctx.winner_team = 'team2'
            self.ctx.loser_team = 'team1'
            self.ctx.game_result = 'team2_win'
        else:
            # This shouldn't happen in a valid game
            self.ctx.winner_team = None
            self.ctx.loser_team = None
            self.ctx.game_result = None
            
        print(f"DEBUG WINNER: first_place={first_place}, winner_team={self.ctx.winner_team}, game_result={self.ctx.game_result}")

    def upgrade(self):
        # Check if we have enough players in win_order
        if len(self.ctx.win_order) < 2:
            return  # Not enough players to determine upgrade
            
        first = self.ctx.win_order[0]
        second = self.ctx.win_order[1]
        
        # Determine winner before calculating upgrade
        self.determine_winner()
        
        # Determine uprank based on partner's position
        # Partner is always (first + 2) % 4
        partner = (first + 2) % 4
        
        # Find partner's position in win_order
        partner_position = None
        for pos, player in enumerate(self.ctx.win_order):
            if player == partner:
                partner_position = pos + 1  # 1-indexed position
                break
        
        if partner_position == 2:  # Partner is Follower (2nd place)
            uprank = 3
        elif partner_position == 3:  # Partner is Third (3rd place)
            uprank = 2
        else:  # Partner is Fourth or worse
            uprank = 1
        # Episode completion check moved to battle_init() to avoid win_order reset timing issue
        if not self.episode_end:     # 这局没有完的时候更新相应的等级
            self.ctx.players[first].playing_myself = True     # 下回合是打谁的level
            self.ctx.players[(first + 2) % 4].playing_myself = True
            self.ctx.players[(first + 1) % 4].playing_myself = False
            self.ctx.players[(first + 3) % 4].playing_myself = False
            
            # Fix Q/K level skip prevention rule
            current_level = self.ctx.players[first].my_rank
            if (current_level == 11 or current_level == 12) and uprank >= 2:  # Q(11) or K(12) level with 2+ promotion
                level = 13  # Must go to A, cannot skip it
            else:
                level = min(13, current_level + uprank)
            
            self.ctx.cur_rank = RANK2[level]
            for i in range(4):
                if i == first or i == (first + 2) % 4:
                    self.ctx.players[i].my_rank = level
                self.ctx.players[i].cur_rank = RANK2[level]

    def one_episode(self):
        self.ctx.cur_rank = '2'
        while not self.episode_end:
            self.deal_cards()
            self.battle_init()
            while not self.round_end:
                self.step()

    def tribute_step(self, card_list, id, action=None, flag=0):     # flag=0表示进贡，flag=1表示还贡
        if not flag:
            legalactions = tribute_legal(card_list, self.ctx)
        else:
            legalactions = back_legal(card_list, self.ctx)
        if action is None:
            # default random action
            if len(legalactions) > 0:
                action = np.random.randint(0, len(legalactions))
            else:
                # No legal actions available, return empty card
                return []
        
        if action < len(legalactions):
            card = legalactions[action]
        else:
            # Action index out of range, return first action or empty
            card = legalactions[0] if legalactions else []

        return card

    def compare(self, card1, card2):     # 比较card1和card2哪个更大
        # Handle empty cards (no tribute available)
        if not card1 and not card2:
            return 0  # Both empty, equal
        elif not card1:
            return -1  # card1 empty, card2 wins
        elif not card2:
            return 1   # card2 empty, card1 wins
        
        # Handle list inputs (convert to single card value)
        if isinstance(card1, list):
            card1 = card1[0] if card1 else -1
        if isinstance(card2, list):
            card2 = card2[0] if card2 else -1
            
        # Handle invalid cards
        if card1 < 0 and card2 < 0:
            return 0
        elif card1 < 0:
            return -1
        elif card2 < 0:
            return 1
            
        if card1 >= 52 and card2 >= 52:
            if card1 > card2:
                return 1
            elif card1 == card2:
                return 0
            else:
                return -1
        elif card1 >= 52 and card2 < 52:
            return 1
        elif card1 < 52 and card2 >= 52:
            return -1
        else:        # 都不是王的情况，如果是级牌则是最大的，不然按数字来
            card1 = 50 if card1 % 13 == CardToNum['H' + self.ctx.cur_rank] else card1 % 13
            card2 = 50 if card2 % 13 == CardToNum['H' + self.ctx.cur_rank] else card2 % 13
            if card1 > card2:
                return 1
            elif card1 == card2:
                return 0
            else:
                return -1

    def tribute_act(self, out, recv, card):   # out拿出card给recv，表示进贡和还贡的单向动作
        # Skip tribute if card is empty (no legal actions)
        if not card:
            return
            
        card_out = deepcopy(self.ctx.players[out].handcards_in_list)
        card_recv = deepcopy(self.ctx.players[recv].handcards_in_list)
        
        # Check if card is actually in the player's hand
        if card not in card_out:
            return
            
        card_out.remove(card)
        card_recv.append(card)
        self.ctx.players[out].update_cards(card_out)
        self.ctx.players[recv].update_cards(card_recv)

    def broadcast_tribute(self):
        pass

    def broadcast_back(self):
        pass

    def broadcast_anti(self):
        pass

    def tribute_phase(self):  # 队友是下游，也要给另一个进贡
        if len(self.ctx.win_order) == 0:
            return
        else:
            card_last = deepcopy(self.ctx.players[self.ctx.win_order[-1]].handcards_in_list)
            card_first = deepcopy(self.ctx.players[self.ctx.win_order[0]].handcards_in_list)
            if abs(self.ctx.win_order[-2] - self.ctx.win_order[-1]) == 2:   # 双下游
                self.anti_tri = anti_tribute(self.ctx, flag=1)
                if self.anti_tri:
                    print('anti tribute')
                    self.broadcast_anti()
                    return
                else:
                    tribute_last = self.tribute_step(card_last, id=self.ctx.win_order[-1])
                    card_third = deepcopy(self.ctx.players[self.ctx.win_order[-2]].handcards_in_list)
                    tribute_third = self.tribute_step(card_third, id=self.ctx.win_order[-2])
                    card_second = deepcopy(self.ctx.players[self.ctx.win_order[1]].handcards_in_list)
                    if self.compare(tribute_last, tribute_third) > 0:  # 下游进贡的更大
                        self.tribute_act(self.ctx.win_order[-1], self.ctx.win_order[0], tribute_last)
                        self.tribute_act(self.ctx.win_order[-2], self.ctx.win_order[1], tribute_third)
                        self.tribute_res.append([self.ctx.win_order[-1], self.ctx.win_order[0], tribute_last])
                        self.tribute_res.append([self.ctx.win_order[-2], self.ctx.win_order[1], tribute_third])
                        self.broadcast_tribute()

                        back_first = self.tribute_step(card_first, id=self.ctx.win_order[0], flag=1)
                        back_second = self.tribute_step(card_second, id=self.ctx.win_order[1], flag=1)
                        self.tribute_act(self.ctx.win_order[0], self.ctx.win_order[-1], back_first)
                        self.tribute_act(self.ctx.win_order[1], self.ctx.win_order[-2], back_second)
                        self.back_res.append([self.ctx.win_order[0], self.ctx.win_order[-1], back_first])
                        self.back_res.append([self.ctx.win_order[1], self.ctx.win_order[-2], back_second])
                        self.broadcast_back()
                        self.ctx.player_waiting = self.ctx.win_order[-1]

                    elif self.compare(tribute_last, tribute_third) < 0:   # 三游进贡的更大
                        self.tribute_act(self.ctx.win_order[-2], self.ctx.win_order[0], tribute_third)
                        self.tribute_act(self.ctx.win_order[-1], self.ctx.win_order[1], tribute_last)
                        self.tribute_res.append([self.ctx.win_order[-2], self.ctx.win_order[0], tribute_third])
                        self.tribute_res.append([self.ctx.win_order[-1], self.ctx.win_order[1], tribute_last])
                        self.broadcast_tribute()

                        back_first = self.tribute_step(card_first, id=self.ctx.win_order[0], flag=1)
                        back_second = self.tribute_step(card_second, id=self.ctx.win_order[1], flag=1)
                        self.tribute_act(self.ctx.win_order[0], self.ctx.win_order[-1], back_first)
                        self.tribute_act(self.ctx.win_order[1], self.ctx.win_order[-2], back_second)
                        self.back_res.append([self.ctx.win_order[0], self.ctx.win_order[-2], back_first])
                        self.back_res.append([self.ctx.win_order[1], self.ctx.win_order[-1], back_second])
                        self.broadcast_back()
                        self.ctx.player_waiting = self.ctx.win_order[-2]

                    else:           # 进贡的大小一样大，按顺时针进贡（给上家进贡），逆时针还贡（给下家还贡）
                        if self.ctx.win_order[-1] == (self.ctx.win_order[0] + 1) % 4:     # 下游是上游的下家
                            self.tribute_act(self.ctx.win_order[-1], self.ctx.win_order[0], tribute_last)
                            self.tribute_act(self.ctx.win_order[-2], self.ctx.win_order[1], tribute_third)
                            self.tribute_res.append([self.ctx.win_order[-1], self.ctx.win_order[0], tribute_last])
                            self.tribute_res.append([self.ctx.win_order[-2], self.ctx.win_order[1], tribute_third])
                            self.broadcast_tribute()

                            back_first = self.tribute_step(card_first, id=self.ctx.win_order[0], flag=1)
                            back_second = self.tribute_step(card_second, id=self.ctx.win_order[1], flag=1)
                            self.tribute_act(self.ctx.win_order[0], self.ctx.win_order[-1], back_first)
                            self.tribute_act(self.ctx.win_order[1], self.ctx.win_order[-2], back_second)
                            self.back_res.append([self.ctx.win_order[0], self.ctx.win_order[-1], back_first])
                            self.back_res.append([self.ctx.win_order[1], self.ctx.win_order[-2], back_second])
                            self.broadcast_back()
                        else:
                            self.tribute_act(self.ctx.win_order[-2], self.ctx.win_order[0], tribute_third)
                            self.tribute_act(self.ctx.win_order[-1], self.ctx.win_order[1], tribute_last)
                            self.tribute_res.append([self.ctx.win_order[-2], self.ctx.win_order[0], tribute_third])
                            self.tribute_res.append([self.ctx.win_order[-1], self.ctx.win_order[1], tribute_last])
                            self.broadcast_tribute()

                            back_first = self.tribute_step(card_first, id=self.ctx.win_order[0], flag=1)
                            back_second = self.tribute_step(card_second, id=self.ctx.win_order[1], flag=1)
                            self.tribute_act(self.ctx.win_order[0], self.ctx.win_order[-1], back_first)
                            self.tribute_act(self.ctx.win_order[1], self.ctx.win_order[-2], back_second)
                            self.back_res.append([self.ctx.win_order[0], self.ctx.win_order[-2], back_first])
                            self.back_res.append([self.ctx.win_order[1], self.ctx.win_order[-1], back_second])
                            self.broadcast_back()
                        self.ctx.player_waiting = (self.ctx.win_order[0] + 1) % 4
                    # print('double tribute', self.tribute_res, self.back_res)
                    # cards_after_tribute = [card_first, card_second, card_third, card_last]
                    # for (index, ele) in enumerate(self.ctx.win_order):
                    #     self.ctx.players[ele].update_cards(cards_after_tribute[index])
                    # for i in range(4):
                    #     print(self.ctx.players[i].handcards_in_dict)
                    #     print(self.ctx.players[i].handcards_in_list)
            else:   # 只有一个下游要进贡
                self.anti_tri = anti_tribute(self.ctx, flag=0)
                if self.anti_tri:
                    print('anti tribute')
                    self.broadcast_anti()
                    return
                else:
                    tribute_last = self.tribute_step(card_last, id=self.ctx.win_order[-1])
                    self.tribute_act(self.ctx.win_order[-1], self.ctx.win_order[0], tribute_last)
                    self.tribute_res.append([self.ctx.win_order[-1], self.ctx.win_order[0], tribute_last])
                    self.broadcast_tribute()

                    back_first = self.tribute_step(card_first, id=self.ctx.win_order[0], flag=1)
                    self.tribute_act(self.ctx.win_order[0], self.ctx.win_order[-1], back_first)
                    self.back_res.append([self.ctx.win_order[0], self.ctx.win_order[-1], back_first])
                    self.broadcast_back()
                    self.ctx.player_waiting = self.ctx.win_order[-1]
                # print('single tribute', self.tribute_res, self.back_res)
                # for i in range(4):
                #     print(self.ctx.players[i].handcards_in_dict)
                #     print(self.ctx.players[i].handcards_in_list)


if __name__ == '__main__':
    ctx = Context()
    game = GameEnv(ctx)
    # game.ctx.cur_rank = '7'
    # game.battle_init()
    # game.tribute_phase()
    # for i in range(4):
    #     print(game.ctx.players[i].handcards_in_dict)
    #     print(game.ctx.players[i].handcards_in_list)
    # print(game.ctx.player_waiting)
    # game.ctx.players[game.ctx.player_waiting].update_cards('H2 H2 S2 C2 C3 D3 H4 S4 S5 C5 H6 H7 S7 H8 H9 H9 C9 D9 HT ST CT HK SK CK CA HA SA')
    # game.step()
    game.one_episode()

