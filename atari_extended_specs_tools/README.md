# Atari extended specs tools

Recommended stress extension:
- 7-game: Breakout Alien Atlantis Boxing Centipede Assault Phoenix
- 8-game (if dataset exists): Breakout Alien Atlantis Boxing Centipede Assault Phoenix Qbert
- Alternative hard candidates: Seaquest, SpaceInvaders, Asterix, BeamRider

Use:
  python scripts/check_atari_dataset_dirs.py --dataset-root /net/tscratch/people/plgdomin088/datasets/atari_expert Assault Phoenix Qbert Seaquest SpaceInvaders
  python scripts/make_atari_extended_specs.py \
    --base configs/specs_atari_cl_5_minari_like_breakout_first.json \
    --dataset-root /net/tscratch/people/plgdomin088/datasets/atari_expert \
    --games Breakout Alien Atlantis Boxing Centipede Assault Phoenix \
    --out configs/specs_atari_cl_7_breakout_first_stress.json
