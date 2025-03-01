from blockworld import BlockWorld
from queue import PriorityQueue

class BlockWorldHeuristic(BlockWorld):
	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)

	def heuristic(self, goal):
		self_state = self.get_state()
		goal_state = goal.get_state()

		total_blocks = sum(len(stack) for stack in goal_state)
		total_matched = 0

		for gstack in goal_state:
			best_match = 0
			for cstack in self_state:
				match = 0
				min_length = min(len(gstack), len(cstack))
				for i in range(1, min_length+1):
					if gstack[-i] == cstack[-i]:
						match += 1
					else:
						break
				if match > best_match:
					best_match = match
			total_matched += best_match

		return total_blocks - total_matched

class AStar():
	def search(self, start, goal):
		# ToDo. Return a list of optimal actions that takes start to goal.
		
		# You can access all actions and neighbors like this:
		# for action, neighbor in state.get_neighbors():
		# 	...

		# Init
		open_queue = PriorityQueue()
		open_queue.put((start.heuristic(goal), start))
		
		g_cost = {start: 0}
		came_from = {start: None}

		closed_set = set()

		# A* search
		while not open_queue.empty():
			f_val, current = open_queue.get()

			# Already visited
			if current in closed_set:
				continue
			closed_set.add(current)

			# Goal state
			if current == goal:
				return self._reconstruct_path(came_from, current)
			
			# Expand
			for action, neighbor in current.get_neighbors():
				tentative_g = g_cost[current] + 1

				if (neighbor not in g_cost) or (tentative_g < g_cost[neighbor]):
					g_cost[neighbor] = tentative_g
					came_from[neighbor] = (current, action)
					f_neighbor = tentative_g + neighbor.heuristic(goal)
					open_queue.put((f_neighbor, neighbor))

		return None


	def _reconstruct_path(self, came_from, current):
		actions = []
		while came_from[current] is not None:
			prev_state, action = came_from[current]
			actions.append(action)
			current = prev_state
		actions.reverse()
		return actions


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