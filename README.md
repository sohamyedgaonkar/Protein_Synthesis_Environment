---
title: Protein Folding Optimization Environment
emoji: "🧬"
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - protein-folding
  - biology
---

# Protein Folding Optimization with Reinforcement Learning

<p align="center">
  <strong>An OpenEnv environment for learning how structural moves can reduce protein energy.</strong>
</p>

<p align="center">
  This project models protein folding as a sequential decision-making problem, where an RL agent learns to reshape a protein chain into lower-energy, more stable conformations.
</p>

---

## Visual Overview

![Protein Folding Banner](../docs/images/banner.png)
<p align="center"><em>Project hero image showing the intersection of molecular structure and intelligent decision making.</em></p>

---

## The Problem We Are Solving

Proteins begin as linear chains of amino acids, but their biological behavior depends on how they fold into stable three-dimensional structures. Even short chains can adopt many possible conformations, and only some of them are energetically favorable.

This environment turns that challenge into a tractable reinforcement learning task:

- A protein is represented as a chain of 3D coordinates.
- The agent applies structural transformations such as torsion rotations and segment flips.
- After every action, the environment recomputes energy and structural quality.
- The agent is rewarded for making the conformation more stable and compact.

In short, the agent is learning this question:

> "What sequence of structural edits most effectively lowers protein energy?"

---

## Why This Matters

Protein folding sits at the center of modern biology, chemistry, and medicine.

- Protein structure determines biological function.
- Misfolded proteins are linked to major diseases.
- Efficient structure optimization supports drug discovery and bioengineering.
- Folding is a natural example of a hard sequential optimization problem.

Even though this project uses a simplified physical model, it captures an important idea:

> local structural changes can produce long-range global effects.

That makes it a strong educational and experimental environment for reinforcement learning.

---

## Why This Problem Is Complex

Protein folding is difficult because it combines:

- high-dimensional geometry
- long-horizon planning
- delayed rewards
- strong physical constraints
- enormous combinational search spaces

One move may improve a local angle but worsen global stability. Another move may temporarily increase disorder before enabling a better final fold. This creates a landscape full of tradeoffs, local minima, and non-obvious action sequences.

![Energy Landscape](../docs/images/energy_landscape.png)
<p align="center"><em>Conceptual energy landscape showing the search for stable low-energy states.</em></p>

---

## Why Reinforcement Learning

Traditional search can evaluate candidate moves step by step, but RL is attractive because the problem is fundamentally sequential.

RL is a strong fit because:

- the agent must act repeatedly over time
- current moves change future possibilities
- rewards are tied to long-term structural quality
- there is no single fixed correct action at each step

Reinforcement learning lets us learn a policy:

> a mapping from protein state to the next structural action

This is especially useful when:

- exact optimization is too expensive
- the action space is large
- outcomes depend on a full action sequence, not isolated decisions

---

## Environment Design

The environment is implemented with OpenEnv and simulates a simplified protein chain.

### Tasks

Three task settings are defined in [openenv.yaml](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/openenv.yaml):

- `task_1`: protein length `8`, goal is to reduce energy by `30%`
- `task_2`: protein length `12`, goal is to form a hydrophobic core
- `task_3`: protein length `20`, goal is to approach minimum energy

### State Representation

Each observation includes:

- `coordinates`: 3D position of each residue
- `torsion_angles`: simplified `[phi, psi]` angles
- `contact_map`: binary residue-residue contact matrix
- `energy`: total conformation energy
- `step_count`: current time step
- `hydrophobic_contacts`: number of favorable hydrophobic interactions
- `collisions`: steric clashes
- `done`: episode termination flag

These fields are defined in [models.py](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/models.py).

### Action Space

The agent can apply structural moves such as:

- `rotate_phi`
- `rotate_psi`
- `pivot_rotation`
- `segment_flip`
- `crankshaft_move`
- `end_move_forward`
- `end_move_backward`

These actions let the policy reshape the protein step by step.

![Structural Actions](../docs/images/actions_diagram.png)
<p align="center"><em>Representative structural edits available to the agent.</em></p>

---

## The Energy Model

The total energy is a sum of simplified physical terms:

```text
E_total =
E_hydrophobic
+ E_steric
+ E_bond
+ E_angle
```

### 1. Hydrophobic Energy

Hydrophobic residues prefer to cluster together.

```text
E_hydrophobic = -1 * hydrophobic_contacts
```

More hydrophobic contacts means lower energy.

### 2. Steric Penalty

Residues should not overlap in space.

```text
E_steric = 5 * collisions
```

Collisions strongly increase energy.

### 3. Bond Constraint

Neighboring residues should preserve approximately correct bond lengths.

```text
E_bond = sum((bond_length - 1.5)^2)
```

### 4. Angle Penalty

Unphysical torsion values are penalized.

```text
E_angle = penalty when torsion angles exceed allowed range
```

All of this logic lives in [my_env_environment.py](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/server/my_env_environment.py).

---

## The Reward System

The reward is shaped to help the agent learn useful folding behavior while still optimizing for energy reduction.

### Reward Components

```text
R_energy = previous_energy - new_energy
R_progress = hydrophobic_contacts_gained
R_stability = +2 if collisions == 0
collision_penalty = -5 * collisions
invalid_action = -10
```

### Final Reward

```text
reward =
2 * R_energy
+ R_progress
+ R_stability
+ collision_penalty
+ invalid_action
```

This reward structure encourages the agent to:

- lower energy
- create favorable hydrophobic packing
- avoid steric clashes
- avoid invalid moves

It is not just optimizing for one number. It is learning structural quality under multiple constraints.

![Reward Flow](../docs/images/reward_calculation.png)
<p align="center"><em>Reward pipeline from action selection to geometric update, energy recomputation, and reward shaping.</em></p>

---

## Why the Reward Design Is Important

A naive reward like "negative energy only" can make training unstable or slow because the agent receives weak guidance. This environment uses reward shaping so the agent gets intermediate signals about whether it is:

- making useful progress
- improving structural stability
- increasing hydrophobic packing
- violating geometry

That makes learning more practical, especially for short proteins and educational RL experiments.

---

## Models in This Project

### 1. Environment Model

The environment model simulates the protein chain and updates:

- coordinates
- torsion angles
- contact maps
- collisions
- total energy

Core file:
- [my_env_environment.py](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/server/my_env_environment.py)

### 2. Search-Based Decision Model

The script [test.py](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/test.py) performs short-horizon search over legal actions and chooses strong moves greedily. It is useful for:

- debugging the environment
- generating interpretable rollouts
- comparing hand-searched decisions against learned policies

### 3. Trained RL Policy

The script [train_policy.py](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/train_policy.py) trains an actor-critic style policy using:

- handcrafted state features
- a linear softmax actor
- a learned value baseline
- reward normalization
- evaluation checkpoints

This is the learned decision-making component of the project.

---

## Training Workflow

### Search-Based Rollout

Use the search harness to inspect strong candidate actions:

```bash
python rl/my_env/test.py --task task_1
```

### Train the Policy

```bash
python rl/my_env/train_policy.py --task task_1 --episodes 400
```

### Evaluate the Best Learned Policy

```bash
python rl/my_env/train_policy.py --mode eval --task task_1 --model-file rl/my_env/models/protein_policy_best.npz
```

### Logs and Metrics

Training metrics are written to:

- [training_metrics.csv](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/logs/training_metrics.csv)

Search and rollout logs are written to:

- [protein_folding_run.log](/c:/Users/DELL/Desktop/tally/meta/Biological/rl/my_env/logs/protein_folding_run.log)

---

## Project Structure

```text
my_env/
|-- README.md
|-- openenv.yaml
|-- models.py
|-- client.py
|-- test.py
|-- train_policy.py
|-- logs/
|   |-- protein_folding_run.log
|   `-- training_metrics.csv
|-- models/
|   |-- protein_policy_best.npz
|   `-- protein_policy_final.npz
`-- server/
    |-- app.py
    `-- my_env_environment.py
```

---

## Images Used

Current image folder:

```text
rl/docs/images/
```

Detected image files:

- `banner.png`
- `energy_landscape.png`
- `actions_diagram.png`
- `reward_calculation.png`

Placement in this README:

- top hero section: `banner.png`
- complexity section: `energy_landscape.png`
- action space section: `actions_diagram.png`
- reward system section: `reward_calculation.png`

---

## In One Sentence

This project uses reinforcement learning to teach an agent how to fold a simplified protein chain by applying structural transformations that reduce energy, improve stability, and encourage hydrophobic packing.
