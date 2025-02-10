from blockworld import BlockWorld
from queue import PriorityQueue

class BlockWorldHeuristic(BlockWorld):
	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)

	def heuristic(self, goal):
		"""
        Enhanced heuristic for BlockWorld.
        
        We start from the idea of counting how many blocks are “out-of-place” via
        a maximum one-to-one matching between the current stacks and the goal stacks.
        That is, let
           h0(s) = (total number of blocks) - (maximum contiguous match sum)
        where the contiguous match is computed from the bottom (since that is how the
        goal is defined).
        
        Then we add two extra penalty terms:
        
         1. Covering penalty: For each current stack, if it already has a contiguous
            (correct) segment from the bottom but then has an extra block on top,
            add 1 penalty unit (since that move “closes down” a block that is already
            placed correctly).
        
         2. Inversion penalty: (Only when the goal is a single stack.) In that case,
            we compute the permutation of blocks (reading bottom-to-top) relative to the
            goal ordering and count how many inversions there are. Each inversion is
            roughly one move away.
        
        The final heuristic is:
        
            h(s) = h0(s) + (covering penalty) + (inversion penalty)
            
        This heuristic is still admissible (each move can fix at most one block’s misplacement
        or one inversion) and it distinguishes “bad” moves (e.g. covering a correctly placed block)
        from “good” moves.
        """
		current_stacks = list(self.get_state())
		goal_stacks = list(goal.get_state())
		total_blocks = sum(len(stack) for stack in current_stacks)

		# ---------------------------
		# Part 1: Maximum contiguous matching from the bottom.
		def contiguous_match(stack1, stack2):
			s1 = list(stack1)[::-1]
			s2 = list(stack2)[::-1]
			common = 0
			for a, b, in zip(s1, s2):
				if a == b:
					common += 1
				else:
					break
			return common
		
		m = len(current_stacks)
		n = len(goal_stacks)
		weights = [[0 for _ in range(n)] for _ in range(m)]
		for i in range(m):
			for j in range(n):
				weights[i][j] = contiguous_match(current_stacks[i], goal_stacks[j])


		def rec(i, used):
			if i == m:
				return 0
			best = rec(i + 1, used)
			for j in range(n):
				if not (used & (1 << j)):
					best = max(best, weights[i][j] + rec(i + 1, used | (1 << j)))
			return best
		
		best_match = rec(0, 0)
		h0 = total_blocks - best_match


		# ---------------------------
		# Part 2: Covering penalty.

		cover_penalty = 0
		for stack in current_stacks:
			s_list = list(stack)
			max_match = 0
			for g_stack in goal_stacks:
				max_match = max(max_match, contiguous_match(stack, g_stack))
			if max_match > 0 and len(s_list) > max_match:
				cover_penalty += 1

		# ---------------------------
		# Part 3: Inversion penalty.
		
		inversion_penalty = 0
		if len(goal_stacks) == 1:
			goal_list = list(goal_stacks[0])
			goal_order = goal_list[::-1]  # bottom first
			mapping = {block: i for i, block in enumerate(goal_order)}
			for stack in current_stacks:
				s_rev = list(stack)[::-1]
				perm = [mapping[b] for b in s_rev if b in mapping]
				inv = 0
				for i in range(len(perm)):
					for j in range(i + 1, len(perm)):
						if perm[i] > perm[j]:
							inv += 1
				inversion_penalty += inv
		

		# ---------------------------
		# Combine the three parts.
		return h0 + cover_penalty + inversion_penalty



class AStar():
	def search(self, start, goal):
		# ToDo. Return a list of optimal actions that takes start to goal.
		
		# You can access all actions and neighbors like this:
		# for action, neighbor in state.get_neighbors():
		# 	...
		"""
        A* search returning a list of actions (each action is a tuple (what, where))
        that brings start to goal. If no solution exists, return None.
        """
		# Initializations
		frontier = PriorityQueue()
		counter = 0 # tie-breaker for items with equal priority
		g_costs = {} # best cost so far for each state
		came_from = {}
		
		# Node 0
		f_start = start.heuristic(goal)
		frontier.put((f_start, counter, start))
		g_costs[start] = 0
		came_from[start] = None

		# A* search
		while not frontier.empty():
			_, _, current = frontier.get()

			if current == goal:
				actions = []
				while came_from[current] is not None:
					prev, action = came_from[current]
					actions.append(action)
					current = prev
				actions.reverse()
				return actions
			
			for action, neighbor in current.get_neighbors():
				tentative_g = g_costs[current] + 1
				if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
					g_costs[neighbor] = tentative_g
					f_neighbor = tentative_g + neighbor.heuristic(goal)
					counter +=1
					frontier.put((f_neighbor, counter, neighbor))
					came_from[neighbor] = (current, action)

		return None

if __name__ == '__main__':
	# Here you can test your algorithm. You can try different N values, e.g. 6, 7.
	#N = 6

	#start = BlockWorldHeuristic(N)
	#goal = BlockWorldHeuristic(N)

	import json
	def load_problem(n, pid):
		with open(f"problems/{n}/{pid}", "r") as f:
			problem = json.load(f)

		return problem

	problem = load_problem(6, 2)
	start = BlockWorldHeuristic(state=problem['start'])
	goal = BlockWorld(state=problem['goal'])

	print("Searching for a path:")
	print(f"{start} -> {goal}")
	print()

	astar = AStar()
	path = astar.search(start, goal)

	if path is not None:
		print("Found a path:")
		print(path)

		print("\nHere's how it goes:")

		s = start.clone()
		print(s)

		for a in path:
			s.apply(a)
			print(s)

	else:
		print("No path exists.")

	print("Total expanded nodes:", BlockWorld.expanded)