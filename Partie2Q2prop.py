# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:03:05 2016

@author: bonndo130
"""

from __future__ import division
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de stratification avec allocations proportionnelles (schéma sans biais)

N=1000000
sigma=0.5
x0=0
K=1
T=1
Beta=0.1#Choisi arbitrairement
n=0
sol=[]
var=[]
A=[]


def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)

while (sps.poisson.pmf(n,T*Beta)*N) >= 1:#On cherche à savoir quand l'allocation proportionnelle ne permet plus d'avoir au moins une simulation 
    A.append(sps.poisson.pmf(n,T*Beta)*N)    
    n = n+1
    
n = n-1#Nombre de strates
A = np.rint(A)#Nombres proportionnels de simulations dans chaque strate 
M = np.sum(A)##Correction d'une éventuelle erreur d'approximation

for i in range (n+1):
    
    psi = []
 
    for j in range (int(A[i])):
        
        unif = [0]
        deltaW = []
    
        for l in range(1,i+1):#Simulation de la statistique d'ordre et des incréments du mouvement brownien 
            
            unif.append(np.random.uniform(0,T,1))
            unif = sorted(unif)
            deltaW.append(np.random.randn(1)*np.sqrt(unif[l]-unif[l-1]))
        
        unif.append(T)
        deltaW.append(np.random.randn(1)*np.sqrt(unif[i+1]-unif[i]))
        
        xchapeau = x0
        prod = 1
        
        if i>0:#Calcul de psi quand N_T>0 
            
            for k in range (1,i+1):
                
                temp = xchapeau        
                xchapeau = xchapeau + u(xchapeau)*(unif[k]-unif[k-1]) + sigma*(deltaW[k-1])
                prod = prod*(u(xchapeau)-u(temp))*(deltaW[k])/(sigma*Beta*(unif[k+1]-unif[k]))
    
            temp = xchapeau    
            xchapeau = xchapeau + u(xchapeau)*(unif[i+1]-unif[i]) + sigma*(deltaW[i])
    
            psi.append(np.exp(Beta*T)*(g(xchapeau)-g(temp))*prod)
    
        else:#Calcul de psi quand N_T=0
            
            xchapeau = xchapeau + u(xchapeau)*(unif[i+1]-unif[i]) + sigma*(deltaW[i])
            
            psi.append(np.exp(Beta*T)*(g(xchapeau)))
            
    sol.append(np.mean(psi)*sps.poisson.pmf(i,T*Beta))    
    var.append(sps.poisson.pmf(i,T*Beta)*np.std(psi)**2)

moyenne6prop = np.sum(sol)
ecartype6prop = np.sqrt(np.sum(var))
bornesup6prop = moyenne6prop + 1.96/np.sqrt(M)*ecartype6prop
borneinf6prop = moyenne6prop - 1.96/np.sqrt(M)*ecartype6prop
diff6prop = bornesup6prop - borneinf6prop

print("Stratification avec allocations proportionnelles (schéma sans biais) :")
print("Estimateur :",moyenne6prop)
print("Ecart-type :",ecartype6prop)
print("Intervalle de confiance à 95% :",[borneinf6prop,bornesup6prop])
print("Erreur :",diff6prop/2)