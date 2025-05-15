import random, time
import math
from collections import defaultdict

import ox

class MCTS:
    """Monte carlo tree seach"""
    def __init__(self, c: float = math.sqrt(2)):
        self.c = c                # exploration weight
        self.Q = defaultdict(int) # total reward of each node
        self.N = defaultdict(int) # number of visits of each node
        self.children = dict()    # board -> {action: child}
        

    def best_action(self, board: ox.Board) -> int:
        """Run MCTS for time_limit seconds -> return best action"""        
        if board not in self.children:
            return random.choice(list(board.get_actions()))
        
        def score(child):
            return (self.Q[child] / self.N[child]) if self.N[child] > 0 else float("-inf")

        best_action, _ = max(
            self.children[board].items(), 
            key = lambda item: score(item[1])
        )
        return best_action

    
    def search(self, board):
        """MCTS tree search"""
        # 1. Selection
        path = self._select(board)
        leaf = path[-1]
        # 2. Expansion
        self._expand(leaf)
        # 3. Simulation
        reward = self._simulate(leaf)
        # 4. Backpropagation
        self._backpropagate(path, reward)
    

    def _select(self, board):
        """Select next move - priority on unexplored nodes, then UCT"""
        path = []
        while True:
            path.append(board)
            if board not in self.children or not self.children[board]:
                return path # Unexplored / terminal board
            
            unexplored = [child for child in self.children[board].values() if child not in self.children]
            if unexplored:
                next_board = unexplored.pop()
                path.append(next_board)
                return path
            board = self._uct_select(board)
    

    def _expand(self, board: ox.Board):
        """Add child note to self.children"""
        if board in self.children:
            return # already expanded
        
        if board.is_terminal():
            self.children[board] = {}
            return
        
        cmap = {}
        for a in board.get_actions():
            child = board.clone()
            child.apply_action(a)
            cmap[a] = child
        self.children[board] = cmap
        

    def _simulate(self, board):
        """Random tree expansion -> get final reward"""
        sim_board = board.clone()
        while not sim_board.is_terminal():
            sim_board.apply_action(random.choice(list(sim_board.get_actions())))
        rewards = sim_board.get_rewards()
        return rewards[board.current_player()]


    def _backpropagate(self, path, reward):
        """Send reward back"""
        for node in reversed(path):
            reward = -reward
            self.N[node] += 1
            self.Q[node] += reward
            


    def _uct_select(self, board):
        """UCT best node selection - balance explotation vs exploration"""
        log_parent = math.log(self.N[board])
        def uct(b):
            return self.Q[b] / self.N[b] + self.c * math.sqrt(log_parent / self.N[b])
        return max(self.children[board].values(), key=uct)


class MCTSBot:
    def __init__(self, play_as: int, time_limit: float):
        self.play_as = play_as
        self.time_limit = time_limit * 0.9
        self.mcts = MCTS()

    def play_action(self, board: ox.Board):
        # TODO: implement MCTS bot
        start_time = time.time()
        while (time.time() - start_time) < self.time_limit:
        #while True:
            self.mcts.search(board)
    
        return self.mcts.best_action(board)
    

if __name__ == '__main__':
    board = ox.Board(8)  # 8x8
    #bots = [MCTSBot(0, 0.1), MCTSBot(1, 1.0)]
    bots = [MCTSBot(0, 10), MCTSBot(1, 12)]

    # try your bot against itself
    while not board.is_terminal():
        current_player = board.current_player()
        current_player_mark = ox.MARKS_AS_CHAR[ ox.PLAYER_TO_MARK[current_player] ]

        current_bot = bots[current_player]
        a = current_bot.play_action(board)
        board.apply_action(a)

        print(f"{current_player_mark}: {a} -> \n{board}\n")
    
    if board.winner is None:
        print("→ It's a draw!")
    elif board.winner == 0:
        print("→ X wins!")
    else:
        print("→ O wins!")
