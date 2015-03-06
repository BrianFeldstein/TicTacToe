# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 07:44:08 2015

@author: Brian
"""

from __future__ import division
import numpy as np
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
from RandomTicValues import GetAvgXRes

#Board is labeled as 012,345,678, for X, and 9..17 fot O, with 1's for filled squares and 0 otherwise.
hidden1 = 30
hidden2 = 30
eps = .4

def sigmoid(z):  #Sigmoid Fixed!!!
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
   
def ValueDV(State, Theta0, Theta1, Theta2, Player): #State is a len 1x18 vector of 0, 1's, (first 9 are X positions, last nine are Os) Theta0 is 19xhidden1, Theta1 is hidden1+1xhidden2 Theta2 is hidden2+1x1
    if IsOver(State)[0]:
        if IsOver(State)[1] == 0: return [.5, 0*Theta0, 0*Theta1, 0*Theta2]        
        if Player == IsOver(State)[1]: return [1, 0*Theta0, 0*Theta1, 0*Theta2]
        if Player == - IsOver(State)[1]: return [0, 0*Theta0, 0*Theta1, 0*Theta2]          
    StateP = np.hstack((np.array([[1]]),State)) 
    z1 = sigmoid(np.dot(StateP, Theta0))
    z1P = np.hstack((np.array([[1]]),z1)) 
    z2 = sigmoid( np.dot(z1P, Theta1) )
    z2P = np.hstack((np.array([[1]]),z2))
    z3 = sigmoid( np.dot(z2P, Theta2) )[0,0]
    
    DVD0 = (z3 - z3**2) *  np.dot( np.transpose(StateP), np.dot(np.transpose(Theta2[1:,:])*(z2 - z2**2), np.transpose(Theta1[1:,:])) * (z1 - z1**2) )
    DVD1 = (z3 - z3**2) * np.dot(np.transpose(z1P), np.transpose(Theta2[1:,0]) * (z2 - z2**2)) 
    DVD2 = (z3 - z3**2) * np.transpose(z2P) 

    if (State[0]==0).all(): return [.5, 0*DVD0, 0*DVD1, 0*DVD2] 
    return [z3, DVD0, DVD1, DVD2]

def BestMove(State, Theta0, Theta1,Theta2, Player, rand, gamefrac): #PLayer is -1 or 1 
    #    State = State18[:, :9] - State18[:, 9:]  
    
    EmptySquaresX = np.where(State[0, :9] == 0)[0]
    EmptySquaresO = np.where(State[0, 9:] == 0)[0]
    EmptySquares = [i for i in EmptySquaresX if i in EmptySquaresO]
    
    if Player == 1:    
        StateList = [np.hstack((State[:,0:i],np.array([[1]]) , State[:,i+1:])) for i in EmptySquares]
    if Player == -1:    
        StateList = [np.hstack((State[:,0:9+i],np.array([[1]]) , State[:,9+i+1:])) for i in EmptySquares]
        
#When random option is chosen, how should theta be adjusted?
    ValueList = [ValueDV(i, Theta0, Theta1, Theta2, Player)[0] for i in StateList]
    if rand:
        x = np.random.random()
        if x > 1:#(np.exp(1-gamefrac) - 1)/(np.e-1): #1*(1-0*gamefrac):
            NextState = StateList[np.argmax(ValueList)]
        else:
            Tot = sum([1 for i in ValueList])    
            ProbList = [1/Tot for i in ValueList]
            prob = rv_discrete(values=(range(len(ProbList)), ProbList))    
            NextState = StateList[prob.rvs()]
    else: NextState = StateList[np.argmax(ValueList)] 
    
    return NextState, True#(NextState == StateList[np.argmax(ValueList)] ).all()
    

def PlayGame(Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, Learner, gamefrac, test = False, pr = False):
    State = np.zeros((1,18))
    Player = 1
    TotTurns = 0
    TotDeltaV = 0    
    
    Continue = True
    while Continue:

        if Player == 1: NewState, IsBest = BestMove(State, Theta0X, Theta1X, Theta2X, Player, Player==Learner or Learner==0, gamefrac)
        if Player == -1: NewState, IsBest = BestMove(State, Theta0O, Theta1O, Theta2O, Player, Player==Learner or Learner==0, gamefrac)
        #if  ValueDV(NewStateO, Theta0X, Theta1X, Player)[0] - ValueDV(StateO, Theta0X, Theta1X, Player)[0] < 0:           

        #if  ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0] !=0:               
        #    OldVal = ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0]
        if test: 
            Ans = GetAvgXRes(State,200)
             
            TotDeltaV += np.abs(ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0] - Ans)      
 
        TotTurns += 1        
        
        if (Learner == 1 or Learner == 0) and IsBest:
            Theta0X = Theta0X + eps*(ValueDV(NewState, Theta0X, Theta1X, Theta2X, 1)[0] - ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[1]          
            Theta1X = Theta1X + eps*(ValueDV(NewState, Theta0X, Theta1X, Theta2X, 1)[0] - ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[2]          
            Theta2X = Theta2X + eps*(ValueDV(NewState, Theta0X, Theta1X, Theta2X, 1)[0] - ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0]) * ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[3]          
     
        if (Learner == -1 or Learner == 0) and IsBest:
            Theta0O = Theta0O + eps*(ValueDV(NewState, Theta0O, Theta1O, Theta2O, -1)[0] - ValueDV(State, Theta0O, Theta1O, Theta2O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, Theta2O, -1)[1]          
            Theta1O = Theta1O + eps*(ValueDV(NewState, Theta0O, Theta1O, Theta2O, -1)[0] - ValueDV(State, Theta0O, Theta1O, Theta2O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, Theta2O, -1)[2]          
            Theta2O = Theta2O + eps*(ValueDV(NewState, Theta0O, Theta1O, Theta2O, -1)[0] - ValueDV(State, Theta0O, Theta1O, Theta2O, -1)[0]) * ValueDV(State, Theta0O, Theta1O, Theta2O, -1)[3]          
        
        #if  ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0] !=0:       
        #    TotDeltaV += np.abs(ValueDV(State, Theta0X, Theta1X, Theta2X, 1)[0] - OldVal)/OldVal
    
        if pr: print [ValueDV(NewState, Theta0X, Theta1X, Theta2X, 1)[0], ValueDV(NewState, Theta0O, Theta1O, Theta2O, -1)[0]] 
            
        Continue = not IsOver(NewState)[0]
            
        State = NewState
        if pr: 
            board = State[:, :9] - State[:, 9:]         
            print board.reshape((3,3))
        Player = Player*(-1)

    return [Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, IsOver(NewState)[1], TotDeltaV/TotTurns]
    
def Learn(GameNum):
    Theta0X = np.random.random((19, hidden1)) - .5
    Theta1X = np.random.random((hidden1+1,hidden2)) - .5
    Theta2X = np.random.random((hidden2+1,1)) - .5
    Theta0O = np.random.random((19, hidden1)) - .5
    Theta1O = np.random.random((hidden1+1,hidden2)) - .5
    Theta2O = np.random.random((hidden2+1,1)) - .5
    A0 = [Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O]
    
    print "go"
    Learner = 0
    AvDVList = []
    TempAvDV = 0
    for i in range(GameNum):
        dotest = False
        if i%100000 < 1000: dotest = True
       
        #Learner = (-1)**(np.floor(i/10000))
        print i
        if i%1000 == 0:
            print i, Learner
            [Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, Result, AvDV] = PlayGame(Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, Learner, i/GameNum,  test = False, pr = True)
            AvDVList.append(TempAvDV/999)
            TempAvDV = 0
            plt.clf()            
            plt.plot(range(len(AvDVList)), AvDVList)
            plt.show()
            plt.pause(1)
            #print "Start Game Numer" + " " + str(i)
        else: [Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, Result, AvDV] = PlayGame(Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, Learner, i/GameNum, test = dotest)
        TempAvDV += AvDV        
        

    PlayGame(Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O, 1, 1, pr = True)
    A = [Theta0X, Theta1X, Theta2X, Theta0O, Theta1O, Theta2O]
    return [A0,A]
        
    
    
    
    
        
