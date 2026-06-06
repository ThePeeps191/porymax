# Porymax - Competitive Pokemon AI & Coaching Platform

A state-of-the-art, recursive reinforcement learning agent for Pokémon Showdown's Gen 9 OverUsed (OU) format. Porymax combines a 142M-parameter transformer policy with Monte Carlo Tree Search (MCTS) lookahead to achieve 1700+ Elo on the official ladder, alongside a built-in Flask web platform for wrappers around the model such as Pokemon Showdown bots, Stockfish-style analysis, and team viability scoring. Porymax also leverages a recursive data flywheel to continuously apply Low-Rank Adaptation (LoRA) and Conservative Q-Learning (CQL) to actively adapt to metagame shifts and new strategies in the Gen 9 OU format.

