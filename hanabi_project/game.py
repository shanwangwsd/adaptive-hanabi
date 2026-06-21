import sys
import copy

from hanabi_core import *

class NullStream:
    def write(self, *args): pass
    def flush(self):        pass


class Game:
    def __init__(self, players, log=sys.stdout, format=0):
        self.players        = players
        for p in self.players:
            if hasattr(p, "reset_episode_state"):
                p.reset_episode_state()
        self.hits           = 3
        self.hints          = 8
        self.current_player = 0
        self.board          = [(c, 0) for c in ALL_COLORS]
        self.played         = []
        self.deck           = make_deck()
        self.extra_turns    = 0
        self.hands          = []
        self.knowledge      = []
        self.trash          = []
        self.log            = log
        self.turn           = 1
        self.format         = format
        self._make_hands()

    def _make_hands(self):
        handsize = 4 if len(self.players) >= 4 else 5
        for i, _ in enumerate(self.players):
            self.hands.append([])
            self.knowledge.append([])
            for _ in range(handsize):
                self._draw_card(i)

    def _draw_card(self, pnr=None):
        if pnr is None: pnr = self.current_player
        if not self.deck: return
        self.hands[pnr].append(self.deck[0])
        self.knowledge[pnr].append(initial_knowledge())
        del self.deck[0]

    def public_state_for(self, observer_pnr):
        """
        Return a public pre-action snapshot for one observer.

        The observer can see every other player's hand, but not their own hand.
        The deck and other hidden internals are intentionally not exposed.
        """
        visible_hands = [
            [] if i == observer_pnr else list(hand)
            for i, hand in enumerate(self.hands)
        ]
        return PublicState(
            hands=visible_hands,
            knowledge=copy.deepcopy(self.knowledge),
            board=copy.deepcopy(self.board),
            trash=list(self.trash),
            played=list(self.played),
            hints=int(self.hints),
            current_player=int(self.current_player),
        )

    def perform(self, action):
        if action not in self.valid_actions():
            raise ValueError(f"Invalid action from player {self.current_player}: {action}")

        for observer_pnr, p in enumerate(self.players):
            p.inform(action, self.current_player, self.public_state_for(observer_pnr))

        if action.type == HINT_COLOR:
            self.hints -= 1
            print(f"{self.players[self.current_player].name} hints "
                  f"{self.players[action.pnr].name} about "
                  f"{COLORNAMES[action.col]} cards  [hints={self.hints}]",
                  file=self.log)
            for (col, num), k in zip(self.hands[action.pnr],
                                     self.knowledge[action.pnr]):
                if col == action.col:
                    for i, row in enumerate(k):
                        if i != col:
                            k[i] = [0] * len(row)
                else:
                    k[action.col] = [0] * len(k[action.col])

        elif action.type == HINT_NUMBER:
            self.hints -= 1
            print(f"{self.players[self.current_player].name} hints "
                  f"{self.players[action.pnr].name} about "
                  f"{action.num}s  [hints={self.hints}]",
                  file=self.log)
            for (col, num), k in zip(self.hands[action.pnr],
                                     self.knowledge[action.pnr]):
                if num == action.num:
                    for row in k:
                        for i in range(len(COUNTS)):
                            if i + 1 != num: row[i] = 0
                else:
                    for row in k:
                        row[action.num - 1] = 0

        elif action.type == PLAY:
            col, num = self.hands[self.current_player][action.cnr]
            if self.board[col][1] == num - 1:
                self.board[col] = (col, num)
                self.played.append((col, num))
                if num == 5:
                    self.hints = min(self.hints + 1, 8)
                print(f"{self.players[self.current_player].name} plays "
                      f"{format_card((col,num))} ✓  board={format_hand(self.board)}",
                      file=self.log)
            else:
                self.trash.append((col, num))
                self.hits -= 1
                print(f"{self.players[self.current_player].name} plays "
                      f"{format_card((col,num))} ✗  hits={self.hits}",
                      file=self.log)
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self._draw_card()

        elif action.type == DISCARD:
            self.hints = min(self.hints + 1, 8)
            card = self.hands[self.current_player][action.cnr]
            self.trash.append(card)
            print(f"{self.players[self.current_player].name} discards "
                  f"{format_card(card)}",
                  file=self.log)
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self._draw_card()

        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def valid_actions(self):
        valid = []
        cp = self.current_player
        for i in range(len(self.hands[cp])):
            valid.append(Action(PLAY,    cnr=i))
            valid.append(Action(DISCARD, cnr=i))
        if self.hints > 0:
            for i, _ in enumerate(self.players):
                if i != cp:
                    for col in set(c for c, _ in self.hands[i]):
                        valid.append(Action(HINT_COLOR,  pnr=i, col=col))
                    for num in set(n for _, n in self.hands[i]):
                        valid.append(Action(HINT_NUMBER, pnr=i, num=num))
        return valid

    def done(self):
        if self.extra_turns == len(self.players) or self.hits == 0:
            return True
        return all(num == 5 for _, num in self.board)

    def score(self):
        return sum(num for _, num in self.board)

    def run(self, turns=-1):
        self.turn = 1
        while not self.done() and (turns < 0 or self.turn < turns):
            self.turn += 1
            if not self.deck:
                self.extra_turns += 1
            hands = [[] if i == self.current_player else h
                     for i, h in enumerate(self.hands)]
            action = self.players[self.current_player].get_action(
                self.current_player, hands, self.knowledge,
                self.trash, self.played, self.board,
                self.valid_actions(), self.hints)
            self.perform(action)
            self.current_player = (self.current_player + 1) % len(self.players)
        print(f"Game done. hits={self.hits} score={self.score()}", file=self.log)
        return self.score()

    def finish(self): pass
