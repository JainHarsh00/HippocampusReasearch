# 🧠 Hippocampus-Inspired Reinforcement Learning Agent

A reinforcement learning project inspired by the computational role of the **hippocampus** in spatial navigation and mental simulation. The project progressively builds an intelligent agent capable of learning a grid-world environment, constructing an internal world model, performing imagination-based planning using the Dyna framework, learning value functions, and adapting its exploration strategy dynamically.

---

## Overview

Traditional reinforcement learning agents learn only through direct interaction with the environment. Biological agents, however, can mentally simulate future experiences before acting.

This project explores that idea by implementing a hippocampus-inspired architecture where the agent:

- Learns a model of the environment
- Simulates future trajectories internally
- Updates action values from imagined experiences
- Chooses actions using planning rather than pure trial-and-error
- Dynamically adjusts exploration based on planning confidence

The implementation is divided into multiple milestones (MVPs), with each notebook introducing one important component.

---

## Project Structure

```
HippocampusProject/
│
├── maze.py                         # Maze environment
├── MVP1_MinimalGridMaze.ipynb      # Grid world environment
├── MVP2_AgentBody.ipynb            # Basic agent interaction
├── MVP3_WorldModel.ipynb           # Internal world model
├── MVP4_DynaPlanning.ipynb         # Planning with imagined rollouts
├── MVP5_ValueFunction.ipynb        # Q-learning integration
├── MVP6_AdaptiveEpsilon.ipynb      # Adaptive exploration strategy
└── README.md
```

---

## Project Progression

### MVP 1 — Minimal Grid Maze

- Creates the grid-world environment
- Defines start, goal, walls, and movement
- Implements environment rendering and interaction

---

### MVP 2 — Agent Body

Introduces an autonomous agent capable of:

- Perceiving the environment
- Taking actions
- Moving through the maze
- Receiving rewards

---

### MVP 3 — World Model

The agent learns an internal representation of the environment by storing:

- State transitions
- Rewards
- Predictive dynamics

This enables the agent to reason about future states without physically visiting them.

---

### MVP 4 — Dyna Planning

Implements Sutton's **Dyna architecture**.

The agent:

- Imagines future trajectories
- Simulates rollouts using its learned model
- Updates knowledge from imagined experiences
- Plans before acting

This mimics hippocampal replay observed in biological navigation.

---

### MVP 5 — Value Function Learning

Adds Q-learning to estimate long-term action values.

Features include:

- Temporal Difference learning
- Bellman updates
- Discounted future rewards
- Improved policy selection

---

### MVP 6 — Adaptive Exploration

Instead of using a fixed exploration rate, the agent dynamically adjusts ε (epsilon) according to planning confidence.

High uncertainty:
- More exploration

High confidence:
- More exploitation

This allows the agent to efficiently balance exploration and exploitation throughout learning.

---

## Key Concepts

- Reinforcement Learning
- Grid World Navigation
- Q-Learning
- Temporal Difference Learning
- World Models
- Model-Based Reinforcement Learning
- Dyna Architecture
- Mental Simulation
- Planning
- Adaptive Exploration
- Computational Neuroscience
- Hippocampus-Inspired Navigation

---

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib jupyter
```

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/yourusername/HippocampusProject.git
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks sequentially:

1. MVP1_MinimalGridMaze.ipynb
2. MVP2_AgentBody.ipynb
3. MVP3_WorldModel.ipynb
4. MVP4_DynaPlanning.ipynb
5. MVP5_ValueFunction.ipynb
6. MVP6_AdaptiveEpsilon.ipynb

Each notebook builds upon the previous milestone.

---

## Future Improvements

Potential extensions include:

- Deep Q Networks (DQN)
- PPO implementation
- Curiosity-driven intrinsic rewards
- Prioritized experience replay
- Continuous control environments
- Partially Observable MDPs
- Transformer-based world models
- MuZero-style planning
- 3D robotic navigation
- ROS2 and Gazebo integration

---

## Applications

This framework can be extended for:

- Autonomous robotics
- Mobile robot navigation
- Space exploration rovers
- Drone path planning
- Computational neuroscience research
- Cognitive architectures
- Model-based AI agents

---

## Inspiration

This project draws inspiration from research in:

- Reinforcement Learning
- Sutton's Dyna Architecture
- Hippocampal Replay
- Place Cells and Grid Cells
- Cognitive Maps
- Model-Based Planning

---

## Author

**Harsh Jain**

B.Tech Artificial Intelligence & Machine Learning

Interests:
- Reinforcement Learning
- Robotics
- Computational Neuroscience
- Autonomous Systems
- Space Robotics
- Artificial Intelligence

---

## License

This project is intended for educational and research purposes.