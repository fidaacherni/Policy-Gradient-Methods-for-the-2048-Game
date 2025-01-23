🎯 Project Overview
This project is a deep reinforcement learning assignment for solving the 2048 game using Policy Gradient methods. The aim is to implement and analyze different policy-based reinforcement learning algorithms, including REINFORCE, REINFORCE with Baseline, and Actor-Critic, leveraging the pgx library for the environment and the equinox framework for neural network implementation.

🛠 Objective
Train a policy to master the 2048 game using policy gradient methods.
Compare the performance of stochastic policies using:
- REINFORCE.
- REINFORCE with Baseline.
- Actor-Critic (with batch sizes of 4 and 200).
Evaluate sampling efficiency, stability and convergence of the algorithms.

🎮 2048 Game Environment
The 2048 game is a grid-based single-player game:
- State Space: A 4×4 grid where tiles with powers of two merge.
- Action Space: Four discrete actions—up, down, left, right.
- Rewards: Accumulated tile values during gameplay.
- Termination: Game ends when no legal moves are available.
The environment is implemented using the pgx library compatible with JAX for efficient computation.

🔍 Implemented Algorithms
REINFORCE:
Implements the vanilla policy gradient.
High variance in updates but simple to implement.

REINFORCE with Baseline:
Introduces a value function as a baseline to reduce gradient variance.
Optimizes both policy and value networks.

Actor-Critic:
Combines policy learning (actor) with value function approximation (critic).
Uses a TD-based advantage estimate to stabilize updates.

🧠 Key Features
Neural Networks:
Custom MLP and CNN models built using equinox.
Actor and Critic networks parameterized for stochastic policies.
Replay Buffers:
Store transitions to compute discounted returns efficiently.
Gradient Computation:
Implements custom loss functions using jax.grad and jax.lax.

📈 Results
REINFORCE:
High variability in episodic returns due to gradient noise.
Requires significant exploration to converge.

REINFORCE with Baseline:
Smoothed gradient updates and faster stabilization.
Moderate improvement over vanilla REINFORCE.
Actor-Critic:
Batch Size 4:
Moderate performance with noticeable fluctuations in returns.
Batch Size 200:
Best overall performance with smooth convergence.

This project highlights the power and challenges of policy gradient methods in reinforcement learning:
- REINFORCE serves as a foundational algorithm.
- Adding a baseline significantly reduces variance and improves stability.
- The Actor-Critic approach achieves the best results, especially with larger batch sizes, showcasing the benefits of blending policy optimization with value estimation.
