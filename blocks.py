import numpy as np


class Block:
    code = None
    color = None
    points = None


class EmptyBlock(Block):
    code = 0


class ObstacleBlock(Block):
    code = 1
    color = (0, 0, 0)
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


class FeedBlock(Block):
    code = 2
    color = (255, 0, 0)
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


class SnakeBlock(Block):
    def __init__(self, front, back):
        self.code = 3 + 5 * front + back
        self.color = (0, 255, 0)
        self.points = self._get_points(front, back)

    @staticmethod
    def decomposed(self, code):
        return (code - 3) / 5, (code - 3) % 5

    @staticmethod
    def _get_points(f, b):
        if (f, b) in [(0, 1), (1, 0)]:
            return np.array([[0.2, 0.8], [1, 0.8], [1, 0.2], [0.8, 0.2], [0.8, 0], [0.2, 0]])
        if (f, b) in [(0, 3), (3, 0)]:
            return np.array([[0.8, 0.8], [0, 0.8], [0, 0.2], [0.2, 0.2], [0.2, 0], [0.8, 0]])
        if (f, b) in [(1, 2), (2, 1)]:
            return np.array([[0.2, 0.2], [1, 0.2], [1, 0.8], [0.8, 0.8], [0.8, 1], [0.2, 1]])
        if (f, b) in [(2, 3), (3, 2)]:
            return np.array([[0.8, 0.2], [0, 0.2], [0, 0.8], [0.2, 0.8], [0.2, 1], [0.8, 1]])
        if (f, b) in [(0, 2), (2, 0)]:
            return np.array([[0.2, 0], [0.8, 0], [0.8, 1], [0.2, 1]])
        if (f, b) in [(1, 3), (1, 3)]:
            return np.array([[0, 0.2], [0, 0.8], [1, 0.8], [1, 0.2]])