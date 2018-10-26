from .minimax import alpha_beta
import random
from collections import OrderedDict

random.seed(45)


class MinMaxAgent:
    def __init__(self, name, depth):
        self.name = name
        self.depth = depth

    def act(self, state, tau):
        action, pi, v1, v2 = self.chose_action(state, tau)
        # new_state, _, _ = state.take_action(action)
        return action, pi, v1, v2

    def chose_action(self, state, tau):
        """

        Args:
            state: state of the board
            tau:  for compatibility with Learning Agent

        Returns:
            best action according to th minmax search
        """
        if tau:
            pass
        best_action = list()
        reward = float('-infinity')
        pi = list()
        for action in state.action_possible:
            new_state, _value, done = state.take_action(action)
            memo = OrderedDict()
            nreward = alpha_beta(new_state, alpha=float('-infinity'), beta=float('infinity'), curr_depth=0,
                                 max_depth=self.depth, memo=memo)
            nreward *= -1
            # print(action, nreward)
            # print(memo)
            # print(OrderedDict(reversed(memo.items())))
            if nreward == reward:
                best_action.append(action)
            elif nreward > reward:
                best_action = [action]
                reward = nreward
            pi.append(nreward)
        # raise ValueError
        return random.choice(best_action), pi, reward, reward
