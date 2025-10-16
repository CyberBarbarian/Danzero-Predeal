from enum import Enum

class ActionType(Enum):
    Passive = 0
    Leading = 1

class Context:
    def __init__(self) -> None:
        self.table = None
        self.card_decks = None
        self.players = None
        self.players_id_list = None
        self.steps = None
        self.player_waiting = None
        self.last_action = None
        self.last_playid = None
        self.last_max_action = None
        self.last_max_playid = None
        self.cur_rank = None
        self.win_order = None
        self.trick_pass = None
        self.recv_wind = None
        self.wind = None
        # Win/loss tracking
        self.winner_team = None  # 'team1' (players 0,2) or 'team2' (players 1,3)
        self.loser_team = None   # 'team1' (players 0,2) or 'team2' (players 1,3)
        self.game_result = None  # 'team1_win', 'team2_win', or None

    def __repr__(self) -> str:
        return 'Context(table={}, players={}, steps={}, player_waiting={}, last_action={})' \
            .format(self.table, self.players, self.steps, self.player_waiting, self.last_action)
