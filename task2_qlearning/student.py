from blockworld import BlockWorldEnv
import random
import time

class QLearning():
	# don't modify the methods' signatures!
	def __init__(self, env: BlockWorldEnv):
		self.env = env

		# Q‑table
		self.Q = {}

		# Hyperparams
		self.alpha = 0.6 # learning‑rate
		self.gamma = 0.9 # discount
		self.epsilon_start = 1.0 # init epsilon
		self.epsilon_end = 0.05 # min epsilon
		self.epsilon_decay_in = 50_000 # total decay steps
		self._steps = 0 # total steps count

	def _current_epsilon(self):
		'''Linear epsilon decay'''
		progress = min(1.0, self._steps / self.epsilon_decay_in)
		return self.epsilon_start * (1.0 - progress) + self.epsilon_end * progress

	def _best_action(self, key, actions):
		"""Choose action with highest Q-value (ties uniformly)."""
		qdict = self.Q.get(key, {})
		
		if not qdict: # unseen state => random
			return random.choice(actions)

		best_q = max(qdict.get(a, 0.0) for a in actions)
		best = [a for a in actions if qdict.get(a, 0.0) == best_q]
		return random.choice(best)


	def train(self):
		# Use BlockWorldEnv to simulate the environment with reset() and step() methods.

		# s = self.env.reset()
		# s_, r, done = self.env.step(a)
		
		# Setup constraints
		time_limit = 29.0
		t_start = time.time()
		max_ep_steps = 50

		while time.time() - t_start < time_limit:
			s, _ = self.env.reset()
			key = (s[0].get_state(), s[1].get_state())

			for _ in range(max_ep_steps):
				actions = s[0].get_actions()

				# Choose A from S (eps-greedy policy)
				if random.random() < self._current_epsilon():
					a = random.choice(actions) # explore
				else:
					a = self._best_action(key, actions) # exploit

				# Interaction
				s_, r, done, _, _ = self.env.step(a)
				key_ = (s_[0].get_state(), s_[1].get_state())

				# Q-table update
				self.Q.setdefault(key, {})
				self.Q.setdefault(key_, {})

				max_next_q = max(self.Q[key_].values(), default=0.0)
				old_q = self.Q[key].get(a, 0.0)
				self.Q[key][a] = old_q + self.alpha * (r + self.gamma * max_next_q - old_q)

				# Move
				s, key = s_, key_
				self._steps += 1
				if done:
					break

		# Exploitation for evaluation
		self.epsilon_start = self.epsilon_end


	def act(self, s):
		key = (s[0].get_state(), s[1].get_state())
		actions = s[0].get_actions()
		
		return self._best_action(key, actions)


if __name__ == '__main__':
	# Here you can test your algorithm. Stick with N <= 4
	N = 4

	env = BlockWorldEnv(N)
	qlearning = QLearning(env)

	# Train
	qlearning.train()

	# Evaluate
	test_env = BlockWorldEnv(N)

	test_problems = 10
	solved = 0
	avg_steps = []

	for test_id in range(test_problems):
		s, _ = test_env.reset()
		done = False

		print(f"\nProblem {test_id}:")
		print(f"{s[0]} -> {s[1]}")

		for step in range(50): 	# max 50 steps per problem
			a = qlearning.act(s)
			s_, r, done, truncated, _ = test_env.step(a)

			print(f"{a}: {s[0]}")

			s = s_

			if done:
				solved += 1
				avg_steps.append(step + 1)
				break

	avg_steps = sum(avg_steps) / len(avg_steps)
	print(f"Solved {solved}/{test_problems} problems, with average number of steps {avg_steps}.")