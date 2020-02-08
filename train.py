import argparse
from dqn_trainer import DQNTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--level_filepath', type=str, required=True, help='level filepath')
parser.add_argument('--episodes', type=int, default=30000, help='the number of episodes to train')
parser.add_argument('--initial_epsilon', type=float, default=1.)
parser.add_argument('--min_epsilon', type=float, default=0.1)
parser.add_argument('--exploration_ratio', type=float, default=0.5)
parser.add_argument('--max_steps', type=int, default=2000)
parser.add_argument('--render_freq', type=int, default=500)
parser.add_argument('--enable_render', type=bool, default=True)
parser.add_argument('--render_fps', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('--enable_save', type=bool, default=True)
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--min_replay_memory_size', type=int, default=1000)
parser.add_argument('--replay_memory_size', type=int, default=100000)
parser.add_argument('--target_update_freq', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--checkpoint', type=str)

args = parser.parse_args()

trainer = DQNTrainer(
    level_filepath=args.level_filepath,
    episodes=args.episodes,
    initial_epsilon=args.initial_epsilon,
    min_epsilon=args.min_epsilon,
    exploration_ratio=args.exploration_ratio,
    max_steps=args.max_steps,
    render_freq=args.render_freq,
    enable_render=args.enable_render,
    render_fps=args.render_fps,
    save_dir=args.save_dir,
    enable_save=args.enable_save,
    save_freq=args.save_freq,
    gamma=args.gamma,
    batch_size=args.batch_size,
    min_replay_memory_size=args.min_replay_memory_size,
    replay_memory_size=args.replay_memory_size,
    target_update_freq=args.target_update_freq,
    seed=args.seed
)

checkpoint = args.checkpoint
if checkpoint is not None:
    trainer.load(checkpoint)

trainer.train()
