# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:36:28 2015

@author: Brian
"""

from __future__ import division
import numpy as np
import itertools

hidden = 30
NumSurvive = 10
OffspringPerParent = 2
NumOffspring =  NumSurvive*OffspringPerParent
NumGames = NumOffspring
alpha = .1

def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
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
    

def Value(State, Theta0, Theta1, Player): #State is a len 1x9 vector of -1, 0, 1's, Theta0 is 10xhidden, Theta1 is hidden+1x1 
    if IsOver(State)[0]:
        if IsOver(State)[1] == 0: return .5       
        if Player == IsOver(State)[1]: return 1
        if Player == - IsOver(State)[1]: return 0         
    StateP = np.hstack((np.array([[1]]),State)) 
    z1 = sigmoid(np.dot(StateP, Theta0))
    z1P = np.hstack((np.array([[1]]),z1)) 
    z2 = sigmoid( np.dot(z1P, Theta1) )[0,0]

    return z2
    
def BestMove(State, Theta0, Theta1, Player): #PLayer is -1 or 1 
    EmptySquares = np.where(State[0] == 0)[0]
    StateList = []
    
    for i in EmptySquares:
        TempState = State.copy()
        TempState[0,i] = Player
        StateList.append(np.array(TempState))
    ValueList = [Value(i, Theta0, Theta1, Player) for i in StateList]
    #print ["%.3f" % i for i in ValueList]    
    NextState = StateList[np.argmax(ValueList)] 
    return NextState
    
    
def PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, pr = False):
    State = np.zeros((1,9))
    Player = 1
    
    Continue = True
    while Continue:

        if Player == 1: NewState = BestMove(State, Theta0X, Theta1X, Player)
        if Player == -1: NewState = BestMove(State, Theta0O, Theta1O, Player)
          
        if pr: print [Value(NewState, Theta0X, Theta1X, 1), Value(NewState, Theta0O, Theta1O, -1)] 
            
        Continue = not IsOver(NewState)[0]
            
        State = NewState
        if pr: print State.reshape((3,3))
        Player = Player*(-1)

    return IsOver(State)[1]
    
def CreateOffspring(Parent, eps):
    ResultList = []
    for i in range(OffspringPerParent):    
        Theta0 = Parent[0] + eps* (np.random.random((10, hidden)) - .5)   
        Theta1 = Parent[1] + eps*(np.random.random((hidden+1, 1)) - .5)
        ResultList.append([Theta0, Theta1])            
    return  ResultList  
    
def Evolve(Generations):
    
    OffspringListX = [ [np.random.random((10, hidden)) - .5, np.random.random((hidden+1, 1)) - .5] for i in range(NumOffspring)]    
    OffspringListO = [ [np.random.random((10, hidden)) - .5, np.random.random((hidden+1, 1)) - .5] for i in range(NumOffspring)]    


    for g in range(Generations):
        #if g%10 == 0: p = True
        #else: p = False
        p = False
        #OpponentsListX = [ np.random.randint(0,NumOffspring, NumGames) for i in range(NumOffspring)]    
        #OpponentsListO = [ np.random.randint(0,NumOffspring, NumGames) for i in range(NumOffspring)]    
    
        OpponentsListX = [ range(NumGames) for i in range(NumOffspring)]    
        OpponentsListO = [ range(NumGames) for i in range(NumOffspring)]    
        
    
        ResultsListX = [ (i, OpponentsListX[i][j], PlayGame(OffspringListX[i][0], OffspringListX[i][1], OffspringListO[OpponentsListX[i][j]][0],OffspringListO[OpponentsListX[i][j]][1], pr= p  )) for i in range(NumOffspring) for j in range(NumGames)]  
        ResultsListO = [ (OpponentsListO[i][j], i, PlayGame(OffspringListX[OpponentsListO[i][j]][0], OffspringListX[OpponentsListO[i][j]][1], OffspringListO[i][0],OffspringListO[i][1], pr= p  )) for i in range(NumOffspring) for j in range(NumGames)]  


        ScoreX = np.zeros(NumOffspring)
        for result in ResultsListX:
            ScoreX[result[0]] += result[2]
        
        ScoreO = np.zeros(NumOffspring)
        for result in ResultsListO:
            ScoreO[result[1]] -= result[2]

            
        SurviveIndicesX = list(ScoreX.argsort()[::-1][0:NumSurvive])
        SurviveIndicesO = list(ScoreO.argsort()[::-1][0:NumSurvive])
        print [ScoreX[i] for i in SurviveIndicesX]
        print [ScoreO[i] for i in SurviveIndicesO]
        SurviveListX = [OffspringListX[i] for i in SurviveIndicesX]
        SurviveListO = [OffspringListO[i] for i in SurviveIndicesO]
        OffspringListX = list(itertools.chain(*[CreateOffspring(j, alpha*(1-g/Generations)) for j in SurviveListX])) 
        OffspringListO = list(itertools.chain(*[CreateOffspring(j, alpha*(1-g/Generations)) for j in SurviveListO])) 
                
        print len([r for r in ResultsListX if r[2] == 0])/len(ResultsListX)
        print len([r for r in ResultsListO if r[2] == 0])/len(ResultsListO)
    
    




  