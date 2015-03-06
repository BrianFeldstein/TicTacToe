# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 20:41:04 2015

@author: Brian
"""

from __future__ import division
import numpy as np

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
    
def PlayTurn(State):
    Player = 1 - 2* sum(State[0])
    EmptySquares = np.where(State[0, :] == 0)[0]
    NewState=State.copy()    
    NewState[0, np.random.choice(EmptySquares)] = Player
    return NewState

def PlayGame(InitState):
    Player = 1 - 2* sum(InitState[0])
    State = InitState.copy()
    while not IsOver(State)[0]:
        State = PlayTurn(State)
        Player *= -1
    return IsOver(State)[1]
        
def GetAvgXRes(InitState, Trials):
    if len(InitState[0]) == 18:
        St = InitState[:, 0:9] - InitState[:,9:]
    else: St = InitState
    Tot = 0
    for i in range(Trials):
        Tot+= (PlayGame(St)+1)/2
    return Tot/Trials
    
    
    
    