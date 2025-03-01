from blockworld import BlockWorld
from queue import PriorityQueue

class BlockWorldHeuristic(BlockWorld):
	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)

	def heuristic(self, goal):
		"""
        Heuristic: count how many blocks are not yet in their correct place.
        
        For each stack in the goal configuration, we compute the longest prefix (starting
        from the bottom of the stack) that appears in some stack of the current state in
        the correct order. The heuristic is then:
        
            h(s) = (total number of blocks) - (sum over goal stacks of best matching length)
            
        This is admissible because each block that is out of place will require at least one move.
        """
		self_state = self.get_state()
		goal_state = goal.get_state()

		total_correct = 0

		for g_stack in goal_state:
			g_list = list(g_stack)
			g_rev = g_list[::-1] # pocitam od spodu hromady

			best_match = 0

			for s_stack in self_state:
				s_list = list(s_stack)
				s_rev = s_list[::-1]

				if not s_rev or not g_rev or s_rev[0] != g_rev[0]:
					continue

				common = 0
				for cur, goal_b in zip(s_rev, g_rev):
					if cur == goal_b:
						common += 1
					else:
						break
				best_match = max(best_match, common)
			total_correct += best_match

		total_blocks = sum(len(stack) for stack in self_state)

		return total_blocks - total_correct


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
		frontier = PriorityQueue()
		counter = 0

		g_costs = {}
		came_from = {}

		g_costs[start] = 0
		
		f_start = start.heuristic(goal)
		frontier.put((f_start, counter, start))
		came_from[start] = None

		while not frontier.empty():
			f_current, _, current = frontier.get()

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
	N = 5

	start = BlockWorldHeuristic(N)
	goal = BlockWorldHeuristic(N)

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