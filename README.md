# Curriculum PPO for SoccerTwos

## Overview

This project implements a **curriculum-based PPO agent with reward shaping** for the SoccerTwos 2v2 environment.

Key contributions:

- **3-phase curriculum learning** (random → moderate → baseline opponent)
- **Dense reward shaping** to address sparse rewards
- **Custom environment wrapper** implementing all reward modifications

Final performance:

- vs Random: **10/10 wins**
- vs Baseline: **9/10 wins (1 draw)**

---

## 🔑 Where to Look (Important for Grading)

### 1. Reward Modification (MAIN REQUIREMENT)

All reward shaping is implemented in: train_three_phase_curriculum.py

Specifically:

- Class: `CurriculumRewardWrapper`
- Function: `_shape_with_stage_logic(...)`

This includes:

- Proximity reward
- Goal progress reward
- Direction alignment
- Ball touch reward
- Pressure / clearance rewards
- Coordination terms (support, off-ball penalty)
- Danger-touch penalty

👉 This directly satisfies the **“Reward Modification (40 pts)”** requirement.

---

### 2. Curriculum Learning Logic

Also implemented in: train_three_phase_curriculum.py

- Opponent switching:
  - Stage 1 → Random
  - Stage 2 → Moderate agent
  - Stage 3 → Baseline agent

- Controlled via:
  - `CurriculumStageSwitchCallback`
  - `_switch_stage_on_all_workers(...)`

---

### 3. Training Script

Main training command:

```bash
python train_three_phase_curriculum.py \
  --phase1-timesteps 2000000 \
  --phase2-timesteps 2500000 \
  --phase3-timesteps 2500000 \
  --num-workers 2 \
  --num-envs-per-worker 1 \
  --experiment-name final_run
```

## SETUP INSTRUCTIONS

### Environment setup

```
conda create -n soccertwos python=3.8 -y
conda activate soccertwos

pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip install -r requirements.txt

pip install protobuf==3.20.3
pip install pydantic==1.10.13
```

### Verify installation

```
python example_random_players.py
```

### Running agent

Watch trained agent vs baseline :

```
python -m soccer_twos.watch -m1 my_agent -m2 ceia_baseline_agent

```

## Repository Structure

.
├── train_three_phase_curriculum.py # MAIN FILE (reward + curriculum)
├── SUBI_FC_AGENT.zip/ # Submitted agent
├── utils.py # Env helpers
├── example_player_agent/ # Baseline example
├── requirements.txt
