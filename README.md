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
<center>
    <img src="/examples/empty.gif" width="250" border=1/>
    <br/>
    <em>9x9 empty</em>
</center>
<center>
    <img src="/examples/obstacles.gif" width="250" border=1/>
    <br/>
    <em>9x9 obstacles</em>
</center>
<center>
    <img src="/examples/double_feed.gif" width="300" border=1/>
    <br/>
    <em>9x13 double feed</em>
</center>
