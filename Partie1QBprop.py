# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
from __future__ import division
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de stratification par rapport au dernier incrément du mouvement brownien avec allocations proportionnelles (Schéma d'Euler)

N=1000000
n=10
sigma=0.5
x0=0
K=1
T=1
delta=T/n
L=10#nombre de strates
sol=[]
var=[]

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)

for i in range (L):
    
    temp = []    
    
    for j in range(N//L):#Simulation des incréments du mouvement brownien dans la strate 
        
        deltaW = np.random.randn(n-1)*np.sqrt(delta)
        deltaW1 = sps.norm.ppf(i/L + np.random.rand()/L,0,np.sqrt(delta))
        deltaW = np.append(deltaW,[deltaW1])
        xdelta = x0
             
        for l in range (n):#Calcul de X_T_delta dans la strate
        
            xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[l])
            
        temp.append(g(xdelta))
    
    var.append(np.std(temp)**2)  
    sol.append(np.mean(temp))
    
moyenne0B = np.mean(sol)
ecartype0B = np.sqrt(np.mean(var))
bornesup0B = moyenne0B + 1.96/np.sqrt(N)*ecartype0B
borneinf0B = moyenne0B - 1.96/np.sqrt(N)*ecartype0B
diff0B = bornesup0B - borneinf0B

print("Stratification dernier incrément brownien allocations proportionnelles (Schéma d'Euler) :")
print("Estimateur :",moyenne0B)
print("Ecart-type :",ecartype0B)
print("Intervalle de confiance à 95% :",[borneinf0B,bornesup0B])
print("Erreur :",diff0B/2)





