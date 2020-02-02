from level_loader import LevelLoader

ll = LevelLoader('levels/9x9_empty.yml')
print(ll.get_field_size())
print(ll.get_field())
print(ll.get_initial_head_position())
print(ll.get_initial_snake())
print(ll.get_initial_tail_position())
print(ll.get_num_feed())