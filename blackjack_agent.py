import random


class BlackjackAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [0, 1]

    def action(self, observation, exploit_only=False) -> int:
        if not exploit_only and random.random() < self.epsilon:
            return random.choice(self.actions)

        else:
            q_values = [self.get_q_value(observation, a) for a in self.actions]
            max_q_value = max(q_values)

            return random.choice([a for a, q in zip(self.actions, q_values) if q == max_q_value])

    def learn(
        self, action, observation, reward, terminated, next_observation
    ) -> float:
        """Returns the current error"""
        old_q_value = self.get_q_value(observation, action)
        next_max_q = max(self.get_q_value(next_observation, a) for a in self.actions)

        # Q-learning update rule
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q - old_q_value)
        self.Q[(observation, action)] = new_q_value
        error = abs(old_q_value - new_q_value)
        return error

    def get_q_value(self, observation, action):
        return self.Q.get((observation, action), 0)