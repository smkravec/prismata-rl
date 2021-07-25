# prismata-rl
Prismata Learning Environment

Prismata is a turn-based perfect information strategy game made by Lunarch studios written in C++. It has a rich action space and a combinatorial like state space from the possibility of many different cards, making a good training ground for model based RL and transfer learning experimentation. The performant codebase and non-visual nature of the game make it very resource efficient.

Uses primsataengine and gym-prismata modules

Currently Built:
- Parallel PPO agent with action masking
- Basic MLP model as well as a dense net architecture (D2RL)
- C++ classical alpha-beta and portfolio MCTS opponents for curricula training or benchmarking
- Capable of loading a neural network opponent for self-play
- GUI for playing against your own agent / testing performance

TO-DO:
*Self-Play league system
*Documentation
*Code refactor / consolidation
*More unit game modes
*Expand MCTS variants
