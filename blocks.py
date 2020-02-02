import numpy as np


class Block:
    @staticmethod
    def contains(**args):
        pass

    @staticmethod
    def get_code(**args):
        pass

    @staticmethod
    def get_color(**args):
        pass

    @staticmethod
    def get_points(**args):
        pass


class EmptyBlock(Block):
    @staticmethod
    def contains(code):
        return code == 0

    @staticmethod
    def get_code():
        return 0


class ObstacleBlock(Block):
    @staticmethod
    def contains(code):
        return code == 1

    @staticmethod
    def get_code():
        return 1

    @staticmethod
    def get_color():
        return 127, 127, 127

    @staticmethod
    def get_points():
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


class FeedBlock(Block):
    @staticmethod
    def contains(code):
        return code == 2

    @staticmethod
    def get_code():
        return 2

    @staticmethod
    def get_color():
        return 255, 0, 102

    @staticmethod
    def get_points():
        return np.array([
            [0.4, 0.2], [0.6, 0.2], [0.8, 0.4], [0.8, 0.6], [0.6, 0.8], [0.4, 0.8], [0.2, 0.6], [0.2, 0.4]
        ])


class SnakeHeadBlock(Block):
    @staticmethod
    def contains(code):
        return 3 <= code < 7

    @staticmethod
    def get_code(d):
        return 3 + d

    @staticmethod
    def get_color():
        return 51, 153, 51

    @staticmethod
    def get_points(code):
        if code == 3:
            return np.array([[0.3, 0.6], [0.7, 0.6], [0.8, 1], [0.2, 1]])
        if code == 4:
            return np.array([[0.4, 0.3], [0.4, 0.7], [0, 0.8], [0, 0.2]])
        if code == 5:
            return np.array([[0.3, 0.4], [0.7, 0.4], [0.8, 0], [0.2, 0]])
        if code == 6:
            return np.array([[0.6, 0.3], [0.6, 0.7], [1, 0.8], [1, 0.2]])


class SnakeBodyBlock(Block):
    @staticmethod
    def contains(code):
        return 7 <= code < 13

    @staticmethod
    def get_code(fd, bd):
        if (fd, bd) in [(0, 3), (1, 2)]:
            return 7
        if (fd, bd) in [(0, 1), (3, 2)]:
            return 8
        if (fd, bd) in [(1, 0), (2, 3)]:
            return 9
        if (fd, bd) in [(2, 1), (3, 0)]:
            return 10
        if (fd, bd) in [(0, 0), (2, 2)]:
            return 11
        if (fd, bd) in [(1, 1), (3, 3)]:
            return 12

    @staticmethod
    def get_color():
        return 0, 204, 102

    @staticmethod
    def get_points(code):
        if code == 7:
            return np.array([[0.2, 0.8], [1, 0.8], [1, 0.2], [0.8, 0.2], [0.8, 0], [0.2, 0]])
        if code == 8:
            return np.array([[0.8, 0.8], [0, 0.8], [0, 0.2], [0.2, 0.2], [0.2, 0], [0.8, 0]])
        if code == 9:
            return np.array([[0.2, 0.2], [1, 0.2], [1, 0.8], [0.8, 0.8], [0.8, 1], [0.2, 1]])
        if code == 10:
            return np.array([[0.8, 0.2], [0, 0.2], [0, 0.8], [0.2, 0.8], [0.2, 1], [0.8, 1]])
        if code == 11:
            return np.array([[0.2, 0], [0.8, 0], [0.8, 1], [0.2, 1]])
        if code == 12:
            return np.array([[0, 0.2], [0, 0.8], [1, 0.8], [1, 0.2]])


class SnakeTailBlock(Block):
    @staticmethod
    def contains(code):
        return 13 <= code < 17

    @staticmethod
    def get_code(d):
        return 13 + d

    @staticmethod
    def get_color():
        return 0, 204, 102

    @staticmethod
    def get_points(code):
        if code == 13:
            return np.array([[0.3, 0.4], [0.7, 0.4], [0.8, 0], [0.2, 0]])
        if code == 14:
            return np.array([[0.6, 0.3], [0.6, 0.7], [1, 0.8], [1, 0.2]])
        if code == 15:
            return np.array([[0.3, 0.6], [0.7, 0.6], [0.8, 1], [0.2, 1]])
        if code == 16:
            return np.array([[0.4, 0.3], [0.4, 0.7], [0, 0.8], [0, 0.2]])


def get_color_points(code):
    if EmptyBlock.contains(code):
        return None
    if ObstacleBlock.contains(code):
        return ObstacleBlock.get_color(), ObstacleBlock.get_points()
    if FeedBlock.contains(code):
        return FeedBlock.get_color(), FeedBlock.get_points()
    if SnakeHeadBlock.contains(code):
        return SnakeHeadBlock.get_color(), SnakeHeadBlock.get_points(code)
    if SnakeBodyBlock.contains(code):
        return SnakeBodyBlock.get_color(), SnakeBodyBlock.get_points(code)
    if SnakeTailBlock.contains(code):
        return SnakeTailBlock.get_color(), SnakeTailBlock.get_points(code)
