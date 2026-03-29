# Reward-Shaped PPO Team Agent

**Agent name:** RewardShapedPPOTeamAgent  
**Class:** `reward_shaped_team_agent.RewardShapedTeamAgent`

## Description

Submission-ready `soccer_twos.AgentInterface` package for a team-level PPO policy trained with reward shaping.

- Expected run id: `PPO_SoccerRewardShaped_72006_00000_0_2026-03-28_02-29-17`
- The agent supports both flattened (`Discrete(27)`) and direct branched (`MultiDiscrete`) action outputs.
- If the model is missing, the package still imports/instantiates and returns valid no-op actions.

## Checkpoint location

The loader resolves checkpoints in this order:

1. `SOCCERTWOS_REWARD_SHAPED_CHECKPOINT` environment variable.
2. `reward_shaped_team_agent/ray_results/**/<run_id>/checkpoint_*/checkpoint-*`
3. `ray_results/**/<run_id>/checkpoint_*/checkpoint-*` (repo root)
4. `~/ray_results/**/<run_id>/checkpoint_*/checkpoint-*`

## Usage

```bash
python -m soccer_twos.watch reward_shaped_team_agent reward_shaped_team_agent
python -m soccer_twos.evaluate reward_shaped_team_agent
```

To force a specific checkpoint:

```bash
export SOCCERTWOS_REWARD_SHAPED_CHECKPOINT=/absolute/path/to/checkpoint-123
```
