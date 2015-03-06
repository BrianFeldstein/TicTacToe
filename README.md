# TicTacToe
Tic-Tac-Toe AI via self play + reinforcement learning + Neural Network

TicTacToe18 uses a one layer neural network, with 18 input features: first 9 0's or 1's to indicate whether board squares contain X's, then 9 more for O's. This is the most up to date, tested version as of 03/06/2015. Learn is the main function to call, with an argument that is the number of games to play for self-training. A graph is output during training to show the average fraction of draws over the last thousand games. If training is successful, this should approach 1. It has been found that training only succeeds for more than about 150,000 training games.

TicTacToe2Layer uses a two layer neural network.

TicTacToeEvolve uses an evolutionary algorithm to train the neural network. Currently not working.
