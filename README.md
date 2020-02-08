# snake-reinforcement-learning

Snake game AI using deep Q-networks

## training

```
python train.py --level_filepath levels/9x9_empty.yml
```
See `python train.py -h` to check what options available.

## playing

```
python play.py --level_filepath levels/9x9_empty.yml --checkpoint best
```
See `python play.py -h` to check what options available.

## some results

9x9 empty:

![](./examples/empty.gif)

9x9 obstacles:

![](./examples/obstacles.gif)

9x13 double feed:

![](./examples/double_feed.gif)
