import argparse
from dqn_trainer import DQNTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--level_filepath', type=str, required=True, help='level filepath')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--max_steps', type=int, default=2000)
parser.add_argument('--render_fps', type=int, default=10)
parser.add_argument('--load_dir', type=str, default='checkpoints')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--image_saving_dir', type=str)

args = parser.parse_args()

trainer = DQNTrainer(
    level_filepath=args.level_filepath,
    max_steps=args.max_steps,
    save_dir=args.load_dir,
    seed=args.seed
)
trainer.load(args.checkpoint)

trainer.preview(
    render_fps=args.render_fps,
    disable_exploration=True,
    save_dir=args.image_saving_dir
)
