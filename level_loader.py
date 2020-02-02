import yaml
from blocks import *


class LevelLoader:
    def __init__(self, level_filepath):
        with open(level_filepath) as f:
            self.level = yaml.safe_load(f)

        s = self.level['field']
        h, w = len(s), len(s[0])
        self.field_size = h, w

        self.field = np.full(self.field_size, EmptyBlock.get_code())
        for i in range(h):
            for j in range(w):
                if s[i][j] == '#':
                    self.field[i, j] = ObstacleBlock.get_code()
                elif s[i][j] == 'T':
                    self.initial_tail_position = i, j

        hx, hy = self.initial_tail_position
        self.initial_snake = []
        while True:
            next_direction, tx, ty = None, None, None
            for d, (dx, dy) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                tx, ty = hx + dx, hy + dy
                if 0 <= tx < h and 0 <= ty < w and s[tx][ty].isdigit() and int(s[tx][ty]) == d:
                    next_direction = d
                    break
            if next_direction is None:
                break

            if len(self.initial_snake) == 0:
                self.field[hx, hy] = SnakeTailBlock.get_code(next_direction)
            else:
                self.field[hx, hy] = SnakeBodyBlock.get_code(next_direction, self.initial_snake[-1])
            self.initial_snake.append(next_direction)
            hx, hy = tx, ty
        self.initial_head_position = hx, hy

        assert len(self.initial_snake) > 1
        self.field[hx, hy] = SnakeHeadBlock.get_code(self.initial_snake[-1])

    def get_field_size(self):
        return self.field_size

    def get_field(self):
        return self.field

    def get_num_feed(self):
        return self.level['num_feed']

    def get_initial_head_position(self):
        return self.initial_head_position

    def get_initial_tail_position(self):
        return self.initial_tail_position

    def get_initial_snake(self):
        return self.initial_snake
