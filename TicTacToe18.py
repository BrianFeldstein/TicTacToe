# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 19:31:12 2015

@author: Brian
"""

from __future__ import division
import numpy as np
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
from RandomTicValues import GetAvgXRes

#Board is labeled as 012,345,678, for X, and 9..17 fot O, with 1's for filled squares and 0 otherwise.
hidden = 75
eps = .5
gamma = 1

def sigmoid(z):  
    return 1/(1 + np.exp( -z))
    
def IsOver(State18):
    State = State18[:, :9] - State18[:, 9:]
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
   
def ValueDV(State, Theta0, Theta1, Player): #State is a len 1x18 vector of 0, 1's, (first 9 are X positions, last nine are Os) Theta0 is 19xhidden, Theta1 is hidden+1x1 
    if IsOver(State)[0]:
        if IsOver(State)[1] == 0: return [.5, 0*Theta0, 0*Theta1]        
        if Player == IsOver(State)[1]: return [1, 0*Theta0, 0*Theta1]
        if Player == - IsOver(State)[1]: return [0, 0*Theta0, 0*Theta1]          
    StateP = np.hstack((np.array([[1]]),State)) 
    z1 = sigmoid(np.dot(StateP, Theta0))
    z1P = np.hstack((np.array([[1]]),z1)) 
    z2 = sigmoid( np.dot(z1P, Theta1) )[0,0]
    DVD0 = (z2 - z2**2)*np.dot(np.transpose(StateP), np.transpose(Theta1[1:,:]) * (z1 - z1**2))
    DVD1 = (z2 - z2**2)* np.transpose(z1P) 

    if (State[0]==0).all(): return [.5, 0*DVD0, 0*DVD1] 
    return [z2, DVD0, DVD1]

def BestMove(State18, Theta0, Theta1, Player, rand, gamefrac): #PLayer is -1 or 1 
    State = State18[:, :9] - State18[:, 9:]  
    EmptySquares = np.where(State[0] == 0)[0]
    StateList = []
    
    for i in EmptySquares:
        TempState = State.copy()
        TempState[0,i] = Player
        Xs = np.where(TempState[0] == 1)[0]
        Os = np.where(TempState[0] == -1)[0]
        TempState18 = np.zeros(18)
        for x in Xs: TempState18[x] = 1
        for o in Os: TempState18[o + 9] = 1
        StateList.append(np.array([TempState18]))
#When random option is chosen, how should theta be adjusted?
    ValueList = [ValueDV(i, Theta0, Theta1, Player)[0] for i in StateList]
    if rand:
        x = np.random.random()
        if x > 1*(1-gamefrac):#(np.exp(1-gamefrac) - 1)/(np.e-1):
            NextState = StateList[np.argmax(ValueList)]
        else:
            Tot = sum([1 for i in ValueList])    
            ProbList = [1/Tot for i in ValueList]
            prob = rv_discrete(values=(range(len(ProbList)), ProbList))    
            NextState = StateList[prob.rvs()]
    else: NextState = StateList[np.argmax(ValueList)] 
    
    return NextState, (NextState == StateList[np.argmax(ValueList)] ).all()
    

def PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, Learner, gamefrac, test = False, pr = False):
    State = np.zeros((1,18))
    Player = 1
    TotTurns = 0
    TotDeltaV = 0    
    
    Continue = True
    while Continue:

        if Player == 1: NewState, IsBest = BestMove(State, Theta0X, Theta1X, Player, Player==Learner or Learner==0, gamefrac)
        if Player == -1: NewState, IsBest = BestMove(State, Theta0O, Theta1O, Player, Player==Learner or Learner==0, gamefrac)
        #if  ValueDV(NewStateO, Theta0X, Theta1X, Player)[0] - ValueDV(StateO, Theta0X, Theta1X, Player)[0] < 0:           
        
        if test and TotTurns in [0,1,2,3,4,5,6,7,8]:
            Ans = GetAvgXRes(State,50)
            TotDeltaV += np.abs(ValueDV(State, Theta0X, Theta1X, 1)[0] - Ans)
        
        #TotDeltaV += np.abs((ValueDV(NewState, Theta0X, Theta1X, 1)[0] - ValueDV(State, Theta0X, Theta1X, 1)[0])) +   np.abs((ValueDV(NewState, Theta0O, Theta1O, -1)[0] - ValueDV(State, Theta0O, Theta1O, -1)[0]))       
        TotTurns += 1        
        
        if (Learner == 1 or Learner == 0) and IsBest and TotTurns in [1,2,3,4,5,6,7,8,9]:
            Theta0X = Theta0X + gamma**(9-TotTurns) * eps*(ValueDV(NewState, Theta0X, Theta1X, 1)[0] - ValueDV(State, Theta0X, Theta1X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, 1)[1]          
            Theta1X = Theta1X + gamma**(9-TotTurns) * eps*(ValueDV(NewState, Theta0X, Theta1X, 1)[0] - ValueDV(State, Theta0X, Theta1X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, 1)[2]          
        if (Learner == -1 or Learner == 0) and IsBest and TotTurns in [1,2,3,4,5,6,7,8,9]:
            Theta0O = Theta0O + gamma**(9-TotTurns) * eps*(ValueDV(NewState, Theta0O, Theta1O, -1)[0] - ValueDV(State, Theta0O, Theta1O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, -1)[1]          
            Theta1O = Theta1O + gamma**(9-TotTurns) * eps*(ValueDV(NewState, Theta0O, Theta1O, -1)[0] - ValueDV(State, Theta0O, Theta1O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, -1)[2]          
        
        #TotDeltaV -= np.abs((ValueDV(NewState, Theta0X, Theta1X, 1)[0] - ValueDV(State, Theta0X, Theta1X, 1)[0])) +   np.abs((ValueDV(NewState, Theta0O, Theta1O, -1)[0] - ValueDV(State, Theta0O, Theta1O, -1)[0]))       
                
        if pr: print [ValueDV(NewState, Theta0X, Theta1X, 1)[0], ValueDV(NewState, Theta0O, Theta1O, -1)[0]] 
            
        Continue = not IsOver(NewState)[0]
            
        State = NewState
        if pr: 
            board = State[:, :9] - State[:, 9:]         
            print board.reshape((3,3))
        Player = Player*(-1)

    return [Theta0X, Theta1X, Theta0O, Theta1O, IsOver(NewState)[1], TotDeltaV/TotTurns]
    
def Learn(GameNum):
    Theta0X = np.random.random((19, hidden)) - .5
    Theta1X = np.random.random((hidden+1,1)) - .5
    Theta0O = np.random.random((19, hidden)) - .5
    Theta1O = np.random.random((hidden+1,1)) - .5
    A0 = [Theta0X, Theta1X, Theta0O, Theta1O]
    #PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, 1, 1, pr = True)
    print "go"
    Learner = 0
    TiesList = []
    AvDVList = []
    NumTies = 0
    TempAvDV = 0
    for i in range(GameNum):
        dotest = False
        if i%10000 < 1000: dotest = False
        #Learner = (-1)**(np.floor(i/10000))
        #print i
        if i%1000 == 0:
            print i, Learner
            [Theta0X, Theta1X, Theta0O, Theta1O, Result, AvDV] = PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, Learner, i/GameNum, test = dotest, pr = False)
            TempAvDV += AvDV            
            if Result == 0: NumTies += 1            
            AvDVList.append(TempAvDV/1000)
            TiesList.append(NumTies/1000)            
            print NumTies/1000            
            TempAvDV = 0
            NumTies = 0
            plt.clf()            
            plt.plot(range(len(TiesList)), TiesList)
            #plt.plot(range(len(AvDVList)), AvDVList)
            plt.show()
            plt.pause(1)
            #print "Start Game Numer" + " " + str(i)
        else: [Theta0X, Theta1X, Theta0O, Theta1O, Result, AvDV] = PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, Learner, i/GameNum, test = dotest)
        TempAvDV += AvDV 
        if Result == 0: NumTies += 1
        

    PlayGame(Theta0X, Theta1X, Theta0O, Theta1O, 1, 1, test = False, pr = True)
    A = [Theta0X, Theta1X, Theta0O, Theta1O]
    return [A0,A, TiesList]
        
    
    
    
    
        
