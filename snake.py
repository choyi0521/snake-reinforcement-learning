import random
from collections import deque
import pygame
from blocks import *
import numpy as np


NUM_CHANNELS = 27
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
INITIAL_FEED = []
INITIAL_TAIL = (4, 1)
INITIAL_SNAKE = [1, 1, 1]
NUM_FEED = 1
DX, DY = [-1, 0, 1, 0], [0, 1, 0, -1]

BLOCK_PIXELS = 30
SCREEN_SIZE = (BOARD_WIDTH*BLOCK_PIXELS, BOARD_HEIGHT*BLOCK_PIXELS)

class SnakeAction:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class SnakeState:
    def __init__(self):
        self.board = np.full((BOARD_HEIGHT, BOARD_WIDTH), EmptyBlock.code)
        self.snake = deque(INITIAL_SNAKE)
        self.tx, self.ty = INITIAL_TAIL
        self.direction = INITIAL_SNAKE[-1]

        self.board[self.tx, self.ty] = SnakeBlock(INITIAL_SNAKE[0], 4).code
        x, y = INITIAL_TAIL
        for i in range(len(INITIAL_SNAKE)-1):
            x += DX[INITIAL_SNAKE[i]]
            y += DY[INITIAL_SNAKE[i]]
            self.board[x, y] = SnakeBlock(INITIAL_SNAKE[i+1], INITIAL_SNAKE[i]).code
        self.hx, self.hy = x + DX[self.direction], y + DY[self.direction]
        self.board[self.hx, self.hy] = SnakeBlock(4, self.direction).code

        for x, y in INITIAL_OBSTACLE:
            self.board[x, y] = ObstacleBlock.code

        for x, y in INITIAL_FEED:
            self.board[x, y] = FeedBlock.code
        for _ in range(NUM_FEED-len(INITIAL_FEED)):
            self._generate_feed()

    def _generate_feed(self):
        empty_blocks = []
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                if self.board[i][j] == EmptyBlock.code:
                    empty_blocks.append((i, j))

        if len(empty_blocks) > 0:
            x, y = random.sample(empty_blocks, 1)[0]
            self.board[x, y] = FeedBlock.code

    def get_length(self):
        return len(self.snake) + 1

    def move_forward(self):
        hx = self.hx + DX[self.direction]
        hy = self.hy + DY[self.direction]
        if hx < 0 or hx >= BOARD_HEIGHT or hy < 0 or hy >= BOARD_WIDTH \
                or self.board[hx][hy] == ObstacleBlock.code \
                or SnakeBlock.is_snake(self.board[hx][hy]) and (hx, hy) != (self.tx, self.ty):
            return -1, True

        is_feed = self.board[hx][hy] == FeedBlock.code

        if not is_feed:
            self.board[self.tx, self.ty] = EmptyBlock.code
            td = self.snake.popleft()
            self.tx += DX[td]
            self.ty += DY[td]
            self.board[self.tx, self.ty] = SnakeBlock(self.snake[0], 4).code

        self.snake.append(self.direction)
        self.board[self.hx, self.hy] = SnakeBlock(self.snake[-1], self.snake[-2]).code
        self.board[hx, hy] = SnakeBlock(4, self.snake[-1]).code
        self.hx, self.hy = hx, hy

        if is_feed:
            self._generate_feed()
            reward = self.get_length()
        else:
            reward = 0

        return reward, False

    def turn_left(self):
        self.direction = (self.direction + 3) % 4
        return self.move_forward()

    def turn_right(self):
        self.direction = (self.direction + 1) % 4
        return self.move_forward()

    def embedded(self):
        return np.eye(NUM_CHANNELS)[self.board]


class Snake:
    ACTIONS = {
        SnakeAction.MOVE_FORWARD: 'move_forward',
        SnakeAction.TURN_LEFT: 'turn_left',
        SnakeAction.TURN_RIGHT: 'turn_right'
    }

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.state = SnakeState()
        return self.state.embedded()

    def step(self, action):
        reward, done = getattr(self.state, Snake.ACTIONS[action])()
        return self.state.embedded(), reward, done

    def quit(self):
        pygame.quit()

    def render(self, fps):
        pygame.display.set_caption('length: {}'.format(self.state.get_length()))
        pygame.event.pump()
        self.screen.fill((255, 255, 255))

        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                block = code_to_block(self.state.board[i][j])
                if block is EmptyBlock:
                    continue
                pygame.draw.polygon(
                    self.screen,
                    block.color,
                    (block.points + [j, i])*BLOCK_PIXELS
                )
        pygame.display.flip()

        self.clock.tick(fps)
