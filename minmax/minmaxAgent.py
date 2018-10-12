from .minimax import alpha_beta
import random


class MinMaxAgent:
    def __init__(self, name, depth):
        self.name = name
        self.depth = depth

    def act(self, state):
        action = self.chose_action(state)
        new_state, _ = state.take_action(action)
        return new_state

    def chose_action(self, state):
        best_action = list()
        reward = float('-infinity')
        for action in state.action_possible:
            new_state, done = state.take_action(action)
            nreward = alpha_beta(new_state, alpha=float('-infinity'), beta=float('infinity'), curr_depth=0,
                                 max_depth=self.depth, memo={})
            if nreward == reward:
                best_action.append(action)
            elif nreward > reward:
                best_action = [action]
                reward = nreward
        return random.choice(best_action)
