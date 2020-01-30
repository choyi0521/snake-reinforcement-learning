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
        self.color = (0, 255, 0) if front == 4 else (127, 255, 127)
        self.points = self._get_points(front, back)

    @staticmethod
    def is_snake(code):
        return code > 2

    @staticmethod
    def decompose(code):
        return (code - 3) // 5, (code - 3) % 5

    @staticmethod
    def _get_points(f, b):
        # snake body
        if (f, b) in [(0, 3), (1, 2)]:
            return np.array([[0.2, 0.8], [1, 0.8], [1, 0.2], [0.8, 0.2], [0.8, 0], [0.2, 0]])
        if (f, b) in [(0, 1), (3, 2)]:
            return np.array([[0.8, 0.8], [0, 0.8], [0, 0.2], [0.2, 0.2], [0.2, 0], [0.8, 0]])
        if (f, b) in [(1, 0), (2, 3)]:
            return np.array([[0.2, 0.2], [1, 0.2], [1, 0.8], [0.8, 0.8], [0.8, 1], [0.2, 1]])
        if (f, b) in [(2, 1), (3, 0)]:
            return np.array([[0.8, 0.2], [0, 0.2], [0, 0.8], [0.2, 0.8], [0.2, 1], [0.8, 1]])
        if (f, b) in [(0, 0), (2, 2)]:
            return np.array([[0.2, 0], [0.8, 0], [0.8, 1], [0.2, 1]])
        if (f, b) in [(1, 1), (3, 3)]:
            return np.array([[0, 0.2], [0, 0.8], [1, 0.8], [1, 0.2]])

        # snake head
        if (f, b) == (4, 0):
            return np.array([[0.3, 0.6], [0.7, 0.6], [0.8, 1], [0.2, 1]])
        if (f, b) == (4, 1):
            return np.array([[0.4, 0.3], [0.4, 0.7], [0, 0.8], [0, 0.2]])
        if (f, b) == (4, 2):
            return np.array([[0.3, 0.4], [0.7, 0.4], [0.8, 0], [0.2, 0]])
        if (f, b) == (4, 3):
            return np.array([[0.6, 0.3], [0.6, 0.7], [1, 0.8], [1, 0.2]])

        # snake tail
        if (f, b) == (2, 4):
            return np.array([[0.3, 0.6], [0.7, 0.6], [0.8, 1], [0.2, 1]])
        if (f, b) == (3, 4):
            return np.array([[0.4, 0.3], [0.4, 0.7], [0, 0.8], [0, 0.2]])
        if (f, b) == (0, 4):
            return np.array([[0.3, 0.4], [0.7, 0.4], [0.8, 0], [0.2, 0]])
        if (f, b) == (1, 4):
            return np.array([[0.6, 0.3], [0.6, 0.7], [1, 0.8], [1, 0.2]])

        assert False


def code_to_block(code):
    if code == 0:
        return EmptyBlock

    if code == 1:
        return ObstacleBlock

    if code == 2:
        return FeedBlock

    # snake block
    f, b = SnakeBlock.decompose(code)
    return SnakeBlock(f, b)
