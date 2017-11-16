# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 22:25:06 2016

@author: Donatien Bonniau
"""
from __future__ import division
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de stratification par rapport au dernier incrément du mouvement brownien avec allocations optimales (Schéma d'Euler)

N=1000000
N2=1000#Nombre de simulations pour estimer les allocations optimales
n=10
sigma=0.5
x0=0
K=1
T=1
delta=T/n
L=10#Nombre de strates
sol=[]
var=[]


def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)

for i in range (L):#Estimation des allocations optimales
    
    temp = []    
    
    for j in range(N2):#Simulation des incréments du mouvement brownien dans la strate 
        
        deltaW = np.random.randn(n-1)*np.sqrt(delta)
        deltaW1 = sps.norm.ppf(i/L + np.random.rand()/L,0,np.sqrt(delta))
        deltaW = np.append(deltaW,[deltaW1])
        xdelta = x0
             
        for l in range (n):#Calcul de X_T_delta dans la strate
        
            xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[l])
            
        temp.append(g(xdelta))
    
    var.append(np.std(temp))   
    
var = var/np.sum(var)
A = np.rint(var*(N-N2*10))#Nombres optimaux de simulations dans chaque strate
M = np.sum(A)#Correction d'une éventuelle erreur d'approximation

var = []

for i in range (L):#Estimation de V_delta
    
    if A[i]!=0:
        
        temp = []
        
        for j in range(int(A[i])):#Simulation des incréments du mouvement brownien dans la strate 
        
            deltaW = np.random.randn(n-1)*np.sqrt(delta)
            deltaW1 = sps.norm.ppf(i/L + np.random.rand()/L,0,np.sqrt(delta))
            deltaW = np.append(deltaW,[deltaW1])
            xdelta = x0
             
            for l in range (n):#Calcul de X_T_delta dans la strate
        
                xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[l])
            
            temp.append(g(xdelta))
    
        sol.append(np.mean(temp))
        var.append(np.std(temp))

moyenne0Bopti = np.mean(sol)
ecartype0Bopti = np.mean(var)
bornesup0Bopti = moyenne0Bopti + 1.96/np.sqrt(M)*ecartype0Bopti
borneinf0Bopti = moyenne0Bopti - 1.96/np.sqrt(M)*ecartype0Bopti
diff0Bopti = bornesup0Bopti - borneinf0Bopti

print("Stratification dernier incrément brownien allocations optimales (Schéma d'Euler) :")
print("Estimateur :",moyenne0Bopti)
print("Ecart-type :",ecartype0Bopti)
print("Intervalle de confiance à 95% :",[borneinf0Bopti,bornesup0Bopti])
print("Erreur :",diff0Bopti/2)