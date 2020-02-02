import random
from collections import deque
import pygame
from blocks import *
import numpy as np


NUM_CHANNELS = 17
NUM_ACTIONS = 3


class SnakeState:
    DX, DY = [-1, 0, 1, 0], [0, 1, 0, -1]

    def __init__(self, field_size, field, num_feed, initial_head_position, initial_tail_position, initial_snake):
        self.field_height, self.field_width = field_size
        self.field = field.copy()
        self.hx, self.hy = initial_head_position
        self.tx, self.ty = initial_tail_position
        self.snake = deque(initial_snake)
        self.direction = initial_snake[-1]

        for _ in range(num_feed):
            self._generate_feed()

    def _generate_feed(self):
        empty_blocks = []
        for i in range(self.field_height):
            for j in range(self.field_width):
                if self.field[i][j] == EmptyBlock.get_code():
                    empty_blocks.append((i, j))

        if len(empty_blocks) > 0:
            x, y = random.sample(empty_blocks, 1)[0]
            self.field[x, y] = FeedBlock.get_code()

    def get_length(self):
        return len(self.snake) + 1

    def move_forward(self):
        hx = self.hx + SnakeState.DX[self.direction]
        hy = self.hy + SnakeState.DY[self.direction]
        if hx < 0 or hx >= self.field_height or hy < 0 or hy >= self.field_width \
                or ObstacleBlock.contains(self.field[hx][hy]) \
                or SnakeBodyBlock.contains(self.field[hx][hy]):
            return -1, True

        is_feed = FeedBlock.contains(self.field[hx][hy])

        if not is_feed:
            self.field[self.tx, self.ty] = EmptyBlock.get_code()
            td = self.snake.popleft()
            self.tx += SnakeState.DX[td]
            self.ty += SnakeState.DY[td]
            self.field[self.tx, self.ty] = SnakeTailBlock.get_code(self.snake[0])

        self.snake.append(self.direction)
        self.field[self.hx, self.hy] = SnakeBodyBlock.get_code(self.snake[-1], self.snake[-2])
        self.field[hx, hy] = SnakeHeadBlock.get_code(self.snake[-1])
        self.hx, self.hy = hx, hy

        if is_feed:
            self._generate_feed()
            return self.get_length(), False

        return 0, False

    def turn_left(self):
        self.direction = (self.direction + 3) % 4
        return self.move_forward()

    def turn_right(self):
        self.direction = (self.direction + 1) % 4
        return self.move_forward()

    def embedded(self):
        return np.eye(NUM_CHANNELS)[self.field]


class SnakeAction:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class Snake:
    ACTIONS = {
        SnakeAction.MOVE_FORWARD: 'move_forward',
        SnakeAction.TURN_LEFT: 'turn_left',
        SnakeAction.TURN_RIGHT: 'turn_right'
    }

    def __init__(self, level_loader, block_pixels=30):
        self.level_loader = level_loader
        self.block_pixels = block_pixels

        self.field_height, self.field_width = self.level_loader.get_field_size()

        pygame.init()
        self.screen = pygame.display.set_mode((
            self.field_width * block_pixels,
            self.field_height * block_pixels
        ))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.state = SnakeState(
            self.level_loader.get_field_size(),
            self.level_loader.get_field(),
            self.level_loader.get_num_feed(),
            self.level_loader.get_initial_head_position(),
            self.level_loader.get_initial_tail_position(),
            self.level_loader.get_initial_snake()
        )
        self.tot_reward = 0
        return self.state.embedded()

    def step(self, action):
        reward, done = getattr(self.state, Snake.ACTIONS[action])()
        self.tot_reward += reward
        return self.state.embedded(), reward, done

    def quit(self):
        pygame.quit()

    def render(self, fps):
        pygame.display.set_caption('length: {}'.format(self.state.get_length()))
        pygame.event.pump()
        self.screen.fill((255, 255, 255))

        for i in range(self.field_height):
            for j in range(self.field_width):
                cp = get_color_points(self.state.field[i][j])
                if cp is None:
                    continue
                pygame.draw.polygon(
                    self.screen,
                    cp[0],
                    (cp[1] + [j, i])*self.block_pixels
                )

        pygame.display.flip()
        self.clock.tick(fps)
