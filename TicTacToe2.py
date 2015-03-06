# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 19:31:12 2015

@author: Brian
"""

from __future__ import division
import numpy as np
from scipy.stats import rv_discrete

#Board is labeled as 012,345,678, and each square has a value -1, 0, 1 for O, nothing, X.
hidden = 30
eps = .5

def sigmoid(z):
    return 1/(1 + np.exp(z))
    
def IsOver(State):
    if State[0,0] == State[0,1] and State[0,1] == State[0,2] and State[0,0] !=0:
        return (True, State[0,0])
    if State[0,3] == State[0,4] and State[0,4] == State[0,5] and State[0,3] !=0:
        return (True, State[0,3])
    if State[0,6] == State[0,7] and State[0,7] == State[0,8] and State[0,6] !=0:
        return (True, State[0,6])
    if State[0,0] == State[0,3] and State[0,3] == State[0,6] and State[0,0] !=0:
        return (True, State[0,0])
    if State[0,1] == State[0,4] and State[0,4] == State[0,7] and State[0,1] !=0:
        return (True, State[0,1])
    if State[0,2] == State[0,5] and State[0,5] == State[0,8] and State[0,2] !=0:
        return (True, State[0,2])
    if State[0,0] == State[0,4] and State[0,4] == State[0,8] and State[0,0] !=0:
        return (True, State[0,0])
    if State[0,2] == State[0,4] and State[0,4] == State[0,6] and State[0,2] !=0:
        return (True, State[0,2])
    if np.array([State[0,i] !=0 for i in range(State.shape[1])]).all():
        return (True, 0)
    else: return (False, 0)
   
def ValueDV(State, Theta0, Theta1, Player): #State is a len 1x9 vector of -1, 0, 1's, Theta0 is 10xhidden, Theta1 is hidden+1x1 
    if IsOver(State)[0]:
        if IsOver(State)[1] == 0: return [.5, 0*Theta0, 0*Theta1]        
        if Player == IsOver(State)[1]: return [1, 0*Theta0, 0*Theta1]
        if Player == - IsOver(State)[1]: return [0, 0*Theta0, 0*Theta1]          
    StateP = np.hstack((np.array([[1]]),State)) 
    z1 = sigmoid(np.dot(StateP, Theta0))
    z1P = np.hstack((np.array([[1]]),z1)) 
    z2 = sigmoid( np.dot(z1P, Theta1) )[0,0]
    DVD0 = (z2**2 - z2)*np.dot(np.transpose(StateP), np.transpose(Theta1[1:,0]) * (z1**2 - z1))
    DVD1 = (z2**2 - z2)* np.transpose(z1P) 

    return [z2, DVD0, DVD1]

def BestMove(State, Theta0, Theta1, Player): #PLayer is -1 or 1 
    EmptySquares = np.where(State[0] == 0)[0]
    StateList = []
    
    for i in EmptySquares:
        TempState = State.copy()
        TempState[0,i] = Player
        StateList.append(np.array(TempState))
    ValueList = [ValueDV(i, Theta0, Theta1, Player)[0] for i in StateList]
    print ["%.3f" % i for i in ValueList]    
    #Tot = sum([i**2 for i in ValueList])    
    #ProbList = [i**2/Tot for i in ValueList]
    #prob = rv_discrete(values=(range(len(ProbList)), ProbList))    
    #NextState = StateList[prob.rvs()]
    NextState = StateList[np.argmax(ValueList)] 
    return NextState
    

def PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, pr = False):
    State = np.zeros((1,9))
    Player = 1
    
    Continue = True
    while Continue:

        if Player == 1: NewState = BestMove(State, Theta0X, Theta1X, Player)
        if Player == -1: NewState = BestMove(State, Theta0O, Theta1O, Player)
        #if  ValueDV(NewStateO, Theta0X, Theta1X, Player)[0] - ValueDV(StateO, Theta0X, Theta1X, Player)[0] < 0:           
        Theta0X = Theta0X + eps*(ValueDV(NewState, Theta0X, Theta1X, 1)[0] - ValueDV(State, Theta0X, Theta1X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, 1)[1]          
        Theta1X = Theta1X + eps*(ValueDV(NewState, Theta0X, Theta1X, 1)[0] - ValueDV(State, Theta0X, Theta1X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, 1)[2]          
        Theta0O = Theta0O + eps*(ValueDV(NewState, Theta0O, Theta1O, -1)[0] - ValueDV(State, Theta0O, Theta1O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, -1)[1]          
        Theta1O = Theta1O + eps*(ValueDV(NewState, Theta0O, Theta1O, -1)[0] - ValueDV(State, Theta0O, Theta1O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, -1)[2]          
            
        if pr: print [ValueDV(NewState, Theta0X, Theta1X, 1)[0], ValueDV(NewState, Theta0O, Theta1O, -1)[0]] 
            
        Continue = not IsOver(NewState)[0]
            
        State = NewState
        if pr: print State.reshape((3,3))
        Player = Player*(-1)

    return [Theta0X, Theta1X, Theta0O, Theta1O]
    
def Learn(GameNum):
    Theta0X = np.random.random((10, hidden)) - .5
    Theta1X = np.random.random((hidden+1,1)) - .5
    Theta0O = np.random.random((10, hidden)) - .5
    Theta1O = np.random.random((hidden+1,1)) - .5
    A0 = [Theta0X, Theta1X, Theta0O, Theta1O]
    PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, pr = True)
    print "go"
    for i in range(GameNum):
        if i%1000 == 0: print i
        #print "Start Game Numer" + " " + str(i)
        [Theta0X, Theta1X, Theta0O, Theta1O] = PlayGame(Theta0X, Theta1X, Theta0O, Theta1O)
        
    PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, pr = True)
    A = [Theta0X, Theta1X, Theta0O, Theta1O]
    return [A0,A]
        
    
    
    
    
        
