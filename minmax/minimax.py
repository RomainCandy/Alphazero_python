import random


def alpha_beta(state, alpha, beta, curr_depth, max_depth, memo):
    if state.is_terminal():
        memo[str(state)] = 123456 - curr_depth
        return 123456 - curr_depth
    elif curr_depth == max_depth:
        memo[str(state)] = state.evaluate()
        return state.evaluate()
    else:
        best = float('-infinity')
        for child in state.next_state():
            v = -1 * alpha_beta(child, -1 * beta, -1 * alpha, curr_depth + 1, max_depth, memo)
            if v > best:
                best = v
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        memo[str(state)] = best
                        return best
        memo[str(state)] = best
        return best


class Game:
    def __init__(self, board):
        self.board = board
        self.player_turn = 1
        self.state = SimpleState(board, self.player_turn)

    def step(self, action):
        next_state = self.state.take_action(action)
        self.state = next_state
        self.player_turn *= -1


class SimpleState:
    def __init__(self, board, player_turn):
        self.board = board
        self.player_turn = player_turn
        self.action_possible = [0*player_turn, 1*player_turn, 2*player_turn]
        self.memo = {}

    def take_action(self, action):
        new_board = self.board.copy()
        new_board.append(action)
        new_state = SimpleState(new_board, -1*self.player_turn)
        if new_state.is_terminal():
            return new_state, 1
        return new_state, 0

    def is_terminal(self):
        return abs(sum((i * x for i, x in enumerate(self.board, 1)))) >= 47
        # return sum(self.board) >= 11

    def next_state(self):
        for action in self.action_possible:
            yield self.take_action(action)[0]
        # return [SimpleState(self.board + [i], -1 * self.player_turn) for i in range(3)]

    def evaluate(self):
        # return self.player_turn * sum(self.board)
        # return int(self.is_terminal())
        return sum((i * x for i, x in enumerate(self.board, 1)))

    def find_best(self):
        b = alpha_beta(self, alpha=float('-infinity'), beta=float('infinity'), curr_depth=0,
                       max_depth=8, memo=self.memo)
        # for key in sorted(self.memo, key=len):
        #     print(key, self.memo[key])
        # print(self.memo[str(self)])
        return b

    def chose(self):
        best_action = list()
        reward = float('-infinity')
        for action in self.action_possible:
            new_state, done = self.take_action(action)
            nreward = new_state.find_best()
            if nreward == reward:
                best_action.append(action)
            elif nreward > reward:
                best_action = [action]
                reward = nreward
        return random.choice(best_action)

    def __str__(self):
        return str(self.board)


env = SimpleState([], 1)
# env.find_best()
for _ in range(15):
    act = env.chose()
    print(len(env.memo))
    env, _ = env.take_action(act)
    print(env, env.evaluate(), len(env.memo))
    if env.is_terminal():
        print(-1 * env.player_turn, 'WIN')
        break
