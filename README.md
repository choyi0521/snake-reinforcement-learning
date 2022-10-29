# snake-reinforcement-learning

Snake game AI using deep Q-networks

A detailed description is provided in the [samsung software membership blog (Korean)](https://infossm.github.io/blog/2020/02/08/snake-dqn/).

## Training

Example:
```
python train.py --level_filepath levels/9x9_empty.yml
```

## Playing

Example:
```
python play.py --level_filepath levels/9x9_empty.yml --checkpoint best
```

## Results

Here are some awesome results.

9x9 empty:

![](./examples/empty.gif)

9x9 obstacles:

![](./examples/obstacles.gif)

9x13 double feed:

![](./examples/double_feed.gif)
