# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:46:20 2016

@author: coicqu13
"""
from __future__ import division
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de stratification avec allocations optimales (schéma sans biais)

N=1000000
N2=1000#Nombre de simulations pour estimer les allocations optimales
sigma=0.5
x0=0
K=1
T=1
Beta=1#Choisi arbitrairement
n=0
sol=[]
var=[]


def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)

while (sps.poisson.pmf(n,T*Beta)*N) >= 1:#Pour avoir le même nombre de strates que l'allocation proportionnelle
    n = n+1
    
n = n-1#Nombre de strates

for i in range (n+1):#Estimation des allocations optimales
    
    psi = []
    
    for j in range (N2):
        
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
            
    var.append(np.std(psi)*sps.poisson.pmf(i,T*Beta))

A = var/np.sum(var)
A = np.rint(A*(N-N2*n))#Nombres optimaux de simulations dans chaque strate
M = np.sum(A)#Correction d'une éventuelle erreur d'approximation

var=[]   

for i in range (n+1):#Estimation de V
    
    if A[i]!=0:
        
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
        var.append(sps.poisson.pmf(i,T*Beta)*np.std(psi))
        
var = np.sum(var)**2
moyenne6opti = np.sum(sol)
ecartype6opti = np.sqrt(var)
bornesup6opti = moyenne6opti + 1.96/np.sqrt(M)*ecartype6opti
borneinf6opti = moyenne6opti - 1.96/np.sqrt(M)*ecartype6opti
diff6opti = bornesup6opti - borneinf6opti

print("Stratification avec allocations optimales (schéma sans biais) :")
print("Estimateur :",moyenne6opti)
print("Ecart-type :",ecartype6opti)
print("Intervalle de confiance à 95% :",[borneinf6opti,bornesup6opti])
print("Erreur :",diff6opti/2)   