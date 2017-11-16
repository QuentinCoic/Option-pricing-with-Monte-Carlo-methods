# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:43:55 2016

@author: coicqu13
"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode classique par un schéma sans biais

N=1000000
sigma=0.5
x0=0
K=1
T=1
sol=[]
Beta=0.1#Choisi arbitrairement

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)


for j in range(N):
    
    xchapeau = x0 
    prod = 1
    compteur = 0
    sumexp = 0
    vecT = [0]
    deltaW = [0]
    
    while sumexp<T:#Simulation de N_T, des T_k et des incréments du mouvement brownien
        
        a = np.random.exponential(1/Beta,1)
        sumexp = sumexp + a
    
        vecT.append(sumexp)
        deltaW.append(np.random.randn(1)*np.sqrt(a)) 
    
        compteur = compteur + 1
        
    compteur = compteur - 1#=N_T
    vecT[compteur+1] = T
    deltaW[compteur+1] = np.random.randn(1)*np.sqrt((T-vecT[compteur]))
    
    if compteur>0:#Calcul de psi quand N_T>0
        
        for i in range (1,compteur+1):
            
            temp = xchapeau        
            xchapeau = xchapeau + u(xchapeau)*(vecT[i]-vecT[i-1]) + sigma*(deltaW[i])
            prod =prod*(u(xchapeau)-u(temp))*(deltaW[i+1])/(sigma*Beta*(vecT[i+1]-vecT[i]))
    
        temp = xchapeau    
        xchapeau = xchapeau + u(xchapeau)*(vecT[compteur+1]-vecT[compteur]) + sigma*(deltaW[compteur+1])
    
        sol.append(np.exp(Beta*T)*(g(xchapeau)-g(temp))*prod)
    
    else:#Calcul de psi quand N_T=0
        
        xchapeau = xchapeau + u(xchapeau)*(vecT[compteur+1]-vecT[compteur]) + sigma*(deltaW[compteur+1])
        
        sol.append(np.exp((Beta)*T)*(g(xchapeau)))
        
        
        
estimateur5 = np.cumsum(sol)/np.linspace(1,N,N)    
moyenne5 = estimateur5[N-1]
ecartype5 = np.std(sol)
bornesup5 = moyenne5 + 1.96/np.sqrt(N)*ecartype5
borneinf5 = moyenne5 - 1.96/np.sqrt(N)*ecartype5
diff5 = bornesup5 - borneinf5

plt.title("Méthode classique par un schéma sans biais")
plt.grid(True)
plt.plot(np.linspace(1,N,N),estimateur5)
plt.xlabel("Nombre de simulations")
plt.ylabel("Estimateur")
plt.axis([0, N, 0.18, 0.22])
plt.show()

print("Méthode classique par un schéma sans biais :")
print("Estimateur :",moyenne5)
print("Ecart-type :",ecartype5)
print("Intervalle de confiance à 95% :",[borneinf5,bornesup5])
print("Erreur :",diff5/2)