# TicTacToe
Tic-Tac-Toe AI via self play + reinforcement learning + Neural Network

TicTacToe18 uses a one layer neural network, with 18 input features: first 9 0's or 1's to indicate whether board squares contain X's, then 9 more for O's. This is the most up to date, tested version as of 03/06/2015.

Learn is the main function to call, with an argument that is the number of games to play for self-training. A graph is output during training to show the average fraction of draws over the last thousand games. If training is successful, this should approach 1. It has been found that training only succeeds for more than about 150,000 training games.  For much more than ~150,000 games, the AI appears to learn to play perfect tic tac toe (although so far I have only tested this by playing it, not by rigorously checking evey possible board).

Learn outputs the network weights, which can be used to play trial games against the AI.  At the moment this is only possible to do by hand, by calling the BestMove function as needed on a given board to see what the AI will do.  A less silly way of playing against the AI is on the to-do list..

X and O get seperate network weights.  The network judges the "value" of any board position.  Each turn, the AI chooses the move which yields the highest value for the next position OR it moves completely randomly.  The probability of making a move randomly starts at 1, and decreases to 0 linearly with the number of games played.  Training occurs via the following principle: if the value of the state after moving is found to be  different than the value of the state before moving, then the value assigned to the previous state must have been in error.  The network weights are then adjusted slightly to change the value of the previous state towards the value of the subsequent state.  The value of a won board is always set to 1, a losing board to 0, and a drawn board to .5.

TicTacToe2Layer uses a two layer neural network.

TicTacToeEvolve uses an evolutionary algorithm to train the neural network. Currently not working.

Test