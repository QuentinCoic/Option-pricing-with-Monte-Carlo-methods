# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:43:55 2016

@author: coicqu13
"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de variable de contrôle (Schéma sans biais)

N=1000000
sigma=0.5
sigmac=sigma**2
x0=0
K=1
T=1
Beta=0.1#Choisi arbitrairement
sol=[]
solbarre=[]

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)
    
def covariance(a,b):#Pour calculer le b optimal
    amean=np.mean(a)
    bmean=np.mean(b)
    
    somme=0
    
    for i in range (len(a)):
        somme = somme + (a[i] -amean)*(b[i]-bmean)
        
    return somme/(len(a))

for j in range(N):
    
    xchapeau = x0 
    prod = 1
    compteur = 0
    sumexp= 0
    vecT = [0]
    deltaW = [0]
    
    while sumexp<T:#Simulation de N_T, des T_k et des incréments du mouvement brownien
        
        a = np.random.exponential(1/Beta,1)
        sumexp = sumexp + a
    
        vecT.append(sumexp)
        deltaW.append(np.random.randn(1)*np.sqrt(a)) 
    
        compteur = compteur +1
        
    compteur = compteur-1#=N_T
    vecT[compteur+1] = T
    deltaW[compteur+1] = np.random.randn(1)*np.sqrt((T-vecT[compteur]))
    
    if compteur>0:#Calcul de psi et de la variable de contrôle quand N_T>0
        
        for i in range (1,compteur+1):
            
            temp = xchapeau        
            xchapeau = xchapeau + u(xchapeau)*(vecT[i]-vecT[i-1]) + sigma*(deltaW[i])
            prod = prod*(u(xchapeau)-u(temp))*(deltaW[i+1])/(sigma*Beta*(vecT[i+1]-vecT[i]))
    
        temp = xchapeau    
        xchapeau = xchapeau + u(xchapeau)*(vecT[compteur+1]-vecT[compteur]) + sigma*(deltaW[compteur+1])
    
        sol.append(np.exp(Beta*T)*(g(xchapeau)-g(temp))*prod)
        
        solbarre.append(g(np.sum(deltaW)*sigma)*np.exp(Beta*T))
    
    else:#Calcul de psi et de la variable de contrôle quand N_T=0
        
        xchapeau = xchapeau + u(xchapeau)*(vecT[compteur+1]-vecT[compteur]) + sigma*(deltaW[compteur+1])
        
        sol.append(np.exp((Beta)*T)*(g(xchapeau)))
        
        solbarre.append(0)
        
b = covariance(sol,solbarre)/covariance(solbarre,solbarre)#b optimal
m = (np.exp(sigmac*T/2)*sps.norm.cdf(-(np.log(K)-sigmac*T)/np.sqrt(sigmac*T))-K*sps.norm.cdf(-(np.log(K))/np.sqrt(sigmac*T)))*np.exp(Beta*T)*(1-sps.poisson.cdf(0,Beta*T))#Espérance de la variable de contrôle
  
Controle = []

for i in range(N):
    Controle.append(sol[i]-b*(solbarre[i]-m))      

estimateur1B = np.cumsum(Controle)/np.linspace(1,N,N)            
moyenne1B = estimateur1B[N-1]
ecartype1B = np.std(Controle)
bornesup1B = moyenne1B + 1.96/np.sqrt(N)*ecartype1B
borneinf1B = moyenne1B - 1.96/np.sqrt(N)*ecartype1B
diff1B = bornesup1B - borneinf1B

plt.title("Variable de contrôle (Schéma sans biais)")
plt.grid(True)
plt.plot(np.linspace(1,N,N),estimateur1B)
plt.xlabel("Nombre de simulations")
plt.ylabel("Estimateur")
plt.axis([0, N, 0.18, 0.22])
plt.show()

print("Variable de contrôle (Schéma sans biais) :")
print("Estimateur :",moyenne1B)
print("Ecart-type :",ecartype1B)
print("Intervalle de confiance à 95% :",[borneinf1B,bornesup1B])
print("Erreur :",diff1B/2)



