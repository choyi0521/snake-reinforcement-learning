import numpy as np
import random
from collections import deque
import cv2


NUM_CHANNELS = 4
NUM_ACTIONS = 3

"""
BOARD_HEIGHT = 15
BOARD_WIDTH = 17

INITIAL_OBSTACLE = []
INITIAL_FEED = [(7, 12)]
INITIAL_SNAKE = [(7, 1), (7, 2), (7, 3), (7, 4)]
INITIAL_DIRECTION = (0, 1)
"""
BOARD_HEIGHT = 9
BOARD_WIDTH = 9

INITIAL_OBSTACLE = []
INITIAL_FEED = [(4, 7)]
INITIAL_SNAKE = [(4, 1), (4, 2), (4, 3), (4, 4)]
INITIAL_DIRECTION = (0, 1)



BLOCK_PIXELS = 20

class Block:
    EMPTY = 0
    OBSTACLE = 1
    FEED = 2
    SNAKE = lambda front, back: 3 + 5 * front + back
    def decompose_snake(self, x):
        return (x-3)/5, (x-3)%5


class SnakeAction:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class SnakeState:
    def __init__(self):
        self.board = np.full((BOARD_HEIGHT, BOARD_WIDTH), Block.EMPTY)
        self.snake = deque(INITIAL_SNAKE)
        self.dx, self.dy = INITIAL_DIRECTION

        for x, y in INITIAL_OBSTACLE:
            self.board[x, y] = Block.OBSTACLE

        for x, y in INITIAL_FEED:
            self.board[x, y] = Block.FEED

        for x, y in INITIAL_SNAKE:
            self.board[x, y] = Block.SNAKE

    def _generate_feed(self):
        empty_blocks = []
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                if self.board[i][j] == Block.EMPTY:
                    empty_blocks.append((i, j))

        if len(empty_blocks) > 0:
            x, y = random.sample(empty_blocks, 1)[0]
            self.board[x, y] = Block.FEED

    def move_forward(self):
        hx, hy = self.snake[-1][0]+self.dx, self.snake[-1][1]+self.dy
        tx, ty = self.snake.popleft()
        self.board[tx, ty] = Block.EMPTY

        if hx < 0 or hx >= BOARD_HEIGHT or hy < 0 or hy >= BOARD_WIDTH \
            or self.board[hx][hy] == Block.OBSTACLE \
            or self.board[hx][hy] == Block.SNAKE:
            self.snake.appendleft((tx, ty))
            self.board[tx, ty] = Block.SNAKE
            return 0, True

        block = self.board[hx][hy]
        self.snake.append((hx, hy))
        self.board[hx, hy] = Block.SNAKE

        if block == Block.EMPTY:
            reward = 0
        else:
            self.snake.appendleft((tx, ty))
            self.board[tx, ty] = Block.SNAKE
            self._generate_feed()
            reward = 1

        return reward, False

    def turn_left(self):
        self.dx, self.dy = -self.dy, self.dx
        return self.move_forward()

    def turn_right(self):
        self.dx, self.dy = self.dy, -self.dx
        return self.move_forward()

    def embedded(self):
        x1 = np.eye(NUM_CHANNELS)[self.board]
        x2 = np.zeros(BOARD_HEIGHT+BOARD_WIDTH+2)
        x2[self.snake[-1][0]] = 1
        x2[BOARD_HEIGHT+self.snake[-1][1]] = 1
        x2[BOARD_HEIGHT+BOARD_WIDTH:] = [self.dx, self.dy]
        return x1, x2


class Snake:
    ACTIONS = {
        SnakeAction.MOVE_FORWARD: 'move_forward',
        SnakeAction.TURN_LEFT: 'turn_left',
        SnakeAction.TURN_RIGHT: 'turn_right'
    }
    BLOCK_BGR = {
        Block.EMPTY: (255, 255, 255),
        Block.OBSTACLE: (0, 0, 0),
        Block.SNAKE: (0, 255, 0),
        Block.FEED: (0, 0, 255)
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = SnakeState()
        self.score = 0

    def step(self, action):
        reward, done = getattr(self.state, Snake.ACTIONS[action])()
        embedded_state = self.state.embedded()
        self.score += reward
        return embedded_state, reward, done

    def render(self, delay):
        # make complete board to be displayed
        board = self.state.board.copy()

        # make scaled image
        img = np.array([[Snake.BLOCK_BGR[block] for block in row] for row in board]).astype(np.uint8)
        img = np.kron(img, np.ones((BLOCK_PIXELS, BLOCK_PIXELS, 1)))

        # display score
        cv2.putText(img, str(self.score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        # display the image
        cv2.imshow('Snake AI', img)
        cv2.waitKey(delay)
