import random
import copy

from hanabi_core import *

class Player:
    def __init__(self, name, pnr):
        self.name = name
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        return random.choice(valid_actions)

    def inform(self, action, player, public_state):
        pass

    def get_explanation(self):
        return self.explanation


class InnerStatePlayer(Player):
    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        possible = [get_possible(k) for k in knowledge[nr]]
        discards = []
        for i, p in enumerate(possible):
            if playable(p, board):   return Action(PLAY, cnr=i)
            if discardable(p, board): discards.append(i)
        if discards:
            return Action(DISCARD, cnr=random.choice(discards))
        playables = [(i, j) for i, h in enumerate(hands) if i != nr
                     for j, (col, n) in enumerate(h) if board[col][1] + 1 == n]
        if playables and hints > 0:
            i, j = playables[0]
            return (Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
                    if random.random() < 0.5
                    else Action(HINT_NUMBER, pnr=i, num=hands[i][j][1]))
        fallback_discards = [Action(DISCARD, cnr=i) for i in range(len(knowledge[nr]))]
        if fallback_discards:
            return random.choice(fallback_discards)
        return valid_actions[0] if valid_actions else Action(DISCARD, cnr=0)


class IntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.pnr  = pnr
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        possible = [get_possible(k) for k in knowledge[nr]]
        result = None
        self.explanation = []

        # 1. Deterministic own play / discard
        for i, p in enumerate(possible):
            if playable(p, board):
                result = Action(PLAY, cnr=i)
        discards = [i for i, p in enumerate(possible) if discardable(p, board)]
        # Token management: discard a known-safe card when hint tokens are near
        # the cap so that future hints remain possible.
        hint_threshold = 7
        if discards and hints >= hint_threshold and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        # 2. Multi-player intentional hint selection.
        # Original pyhanabi code used 1-nr, which only works for 2 players.
        if hints > 0 and not result:
            valid = []
            for target in range(len(hands)):
                if target == nr or not hands[target]:
                    continue
                if target >= len(knowledge) or not knowledge[target]:
                    continue

                intentions = [None] * len(hands[target])
                for j, (col, num) in enumerate(hands[target]):
                    if board[col][1] + 1 == num:
                        intentions[j] = PLAY
                    elif board[col][1] >= num:
                        intentions[j] = DISCARD
                    elif num < 5:
                        intentions[j] = CANDISCARD

                for c in ALL_COLORS:
                    act = (HINT_COLOR, c)
                    try:
                        ok, sc, _ = pretend(act, knowledge[target], intentions,
                                            hands[target], board)
                    except Exception:
                        continue
                    if ok:
                        valid.append((target, act, sc))
                for r in range(1, 6):
                    act = (HINT_NUMBER, r)
                    try:
                        ok, sc, _ = pretend(act, knowledge[target], intentions,
                                            hands[target], board)
                    except Exception:
                        continue
                    if ok:
                        valid.append((target, act, sc))

            if valid:
                valid.sort(key=lambda x: -x[2])
                target, a, _ = valid[0]
                result = (Action(HINT_COLOR, pnr=target, col=a[1])
                          if a[0] == HINT_COLOR
                          else Action(HINT_NUMBER, pnr=target, num=a[1]))

        if result:
            return result

        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        if not scores:
            return valid_actions[0] if valid_actions else Action(DISCARD, cnr=0)
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def inform(self, action, player, public_state):
        pass


def shift_hint_memory_after_card_removed(hints, player, removed_index):
    """Shift per-card hint memory after a player plays/discards one card.

    The memory key is (card_index, player_id).  Only keys belonging to the
    acting player are shifted, which keeps 4-player hint memories isolated.
    """
    if removed_index is None:
        return

    player_keys = sorted(
        [idx for (idx, pid) in hints.keys() if pid == player and idx >= removed_index]
    )
    if not player_keys:
        return

    old_for_player = {idx: hints.get((idx, player), []) for idx in player_keys}
    for idx in player_keys:
        hints.pop((idx, player), None)

    max_idx = max(player_keys)
    for idx in range(removed_index, max_idx + 1):
        # Preserve explicit empty-list memories.  In OuterStatePlayer and
        # SelfIntentionalPlayer, [] can mean "this card position has been
        # tracked and no further hint type should be repeated"; dropping it
        # would make a shifted card look completely unseen.
        if idx + 1 in old_for_player:
            hints[(idx, player)] = old_for_player[idx + 1]


class OuterStatePlayer(Player):
    """
    Baseline from the original pyhanabi source.
    It uses visible outer-state information: if it sees a teammate has a playable card,
    it gives a color/rank hint for that card, while avoiding repeating the same hint type
    for the same card position.
    """
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.hints = {}
        self.explanation = []

    def reset_episode_state(self):
        self.hints = {}
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        possible = [get_possible(k) for k in knowledge[nr]]

        discards = []
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(hands):
            if i == nr:
                continue
            for j, (col, n) in enumerate(h):
                if board[col][1] + 1 == n:
                    playables.append((i, j))
        playables.sort(key=lambda ij: -hands[ij[0]][ij[1]][1])

        while playables and hints > 0:
            i, j = playables[0]
            hint_types = [HINT_COLOR, HINT_NUMBER]
            if (j, i) not in self.hints:
                self.hints[(j, i)] = []
            for old_hint in self.hints[(j, i)]:
                if old_hint in hint_types:
                    hint_types.remove(old_hint)

            if hint_types:
                chosen = random.choice(hint_types)
                self.hints[(j, i)].append(chosen)
                if chosen == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
                return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])

            playables = playables[1:]

        for i, k in enumerate(knowledge):
            if i == nr or not hands[i]:
                continue
            cards = list(range(len(k)))
            random.shuffle(cards)
            c = cards[0]
            col, num = hands[i][c]
            hint_types = [HINT_COLOR, HINT_NUMBER]
            if (c, i) not in self.hints:
                self.hints[(c, i)] = []
            for old_hint in self.hints[(c, i)]:
                if old_hint in hint_types:
                    hint_types.remove(old_hint)
            if hint_types and hints > 0:
                chosen = random.choice(hint_types)
                self.hints[(c, i)].append(chosen)
                if chosen == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=col)
                return Action(HINT_NUMBER, pnr=i, num=num)

        fallback_discards = [Action(DISCARD, cnr=i) for i in range(handsize)]
        if fallback_discards:
            return random.choice(fallback_discards)
        return valid_actions[0] if valid_actions else Action(DISCARD, cnr=0)

    def inform(self, action, player, public_state):
        if action.type in [PLAY, DISCARD]:
            shift_hint_memory_after_card_removed(self.hints, player, action.cnr)


class SelfIntentionalPlayer(Player):
    """
    Baseline from the original pyhanabi source.
    It extends IntentionalPlayer by first interpreting hints received by itself:
    a pointed card that is potentially playable is treated as intended PLAY;
    a pointed card that is potentially discardable is treated as intended DISCARD.
    """
    def __init__(self, name, pnr):
        self.name = name
        self.pnr = pnr
        self.hints = {}
        self.gothint = None
        self.last_knowledge = []
        self.last_board = []
        self.last_trash = []
        self.last_played = []
        self.explanation = []

    def reset_episode_state(self):
        self.hints = {}
        self.gothint = None
        self.last_knowledge = []
        self.last_board = []
        self.last_trash = []
        self.last_played = []
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board,
                   valid_actions, hints):
        handsize = len(knowledge[nr])
        result = None

        hinted_actions = []
        if self.gothint:
            act, _ = self.gothint
            if act.type == HINT_COLOR:
                for k in knowledge[nr]:
                    hinted_actions.append(whattodo(k, sum(k[act.col]) > 0, board))
            elif act.type == HINT_NUMBER:
                for k in knowledge[nr]:
                    cnt = 0
                    for c in ALL_COLORS:
                        cnt += k[c][act.num - 1]
                    hinted_actions.append(whattodo(k, cnt > 0, board))

        if hinted_actions:
            for i, a in enumerate(hinted_actions):
                if a == PLAY and (not result or result.type == DISCARD):
                    result = Action(PLAY, cnr=i)
                elif a == DISCARD and not result:
                    result = Action(DISCARD, cnr=i)

        self.gothint = None

        possible = [get_possible(k) for k in knowledge[nr]]
        discards = []
        for i, p in enumerate(possible):
            if playable(p, board) and not result:
                result = Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        # Token management: discard a known-safe card when hint tokens are near
        # the cap so that future hints remain possible.
        hint_threshold = 7
        if discards and hints >= hint_threshold and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        if hints > 0 and not result:
            othercards = trash + played
            valid = []
            for target in range(len(hands)):
                if target == nr or not hands[target]:
                    continue
                if target >= len(knowledge) or not knowledge[target]:
                    continue

                intentions = [None] * len(hands[target])
                for j, (col, n) in enumerate(hands[target]):
                    if board[col][1] + 1 == n:
                        intentions[j] = PLAY
                    elif board[col][1] >= n:
                        intentions[j] = DISCARD
                    elif n < 5 and (col, n) not in othercards:
                        intentions[j] = CANDISCARD

                for c in ALL_COLORS:
                    act = (HINT_COLOR, c)
                    try:
                        ok, sc, _ = pretend(act, knowledge[target], intentions,
                                            hands[target], board)
                    except Exception:
                        continue
                    if ok:
                        valid.append((target, act, sc))
                for r in range(1, 6):
                    act = (HINT_NUMBER, r)
                    try:
                        ok, sc, _ = pretend(act, knowledge[target], intentions,
                                            hands[target], board)
                    except Exception:
                        continue
                    if ok:
                        valid.append((target, act, sc))

            if valid:
                valid.sort(key=lambda x: -x[2])
                target, a, _ = valid[0]
                if a[0] == HINT_COLOR:
                    result = Action(HINT_COLOR, pnr=target, col=a[1])
                else:
                    result = Action(HINT_NUMBER, pnr=target, num=a[1])

        if result:
            return result

        scores = [pretend_discard(Action(DISCARD, cnr=i), knowledge[nr], board, trash)
                  for i in range(handsize)]
        if not scores:
            return valid_actions[0] if valid_actions else Action(DISCARD, cnr=0)
        scores.sort(key=lambda x: -x[1])
        return scores[0][0]

    def inform(self, action, player, public_state):
        if action.type in [PLAY, DISCARD]:
            shift_hint_memory_after_card_removed(self.hints, player, action.cnr)
        elif action.type in [HINT_COLOR, HINT_NUMBER] and action.pnr == self.pnr:
            self.gothint = (action, player)
            self.last_knowledge = copy.deepcopy(public_state.knowledge)
            self.last_board = copy.deepcopy(public_state.board)
            self.last_trash = list(public_state.trash)
            self.last_played = list(public_state.played)
