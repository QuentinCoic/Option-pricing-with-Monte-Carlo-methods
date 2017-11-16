# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de fonction d'importance avec un alpha arbitraire simple (Schéma d'Euler)

N=1000000
n=10
sigma=0.5
x0=0
K=1
T=1
delta=T/n
alpharb=1.15#Valeur de alpha choisie arbitrairement
sol=[]

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)
    
def importance(x):#Quotient des deux densités
    return np.exp(-alpharb*np.sum(x)+alpharb**2/2)
    
for i in range (N):
    
    deltaW = np.random.randn(n)*np.sqrt(delta) + delta*alpharb#Simulation des incréments du mouvement brownien translatés de alpha*delta
    xdelta = x0
   
    for j in range (n):#Calcul de X_T_delta avec brownien translaté
            
        xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[j])
        
    sol.append(g(xdelta)*importance(deltaW))

ecartype4 = np.std(sol)
sol = np.cumsum(sol)/np.linspace(1,N,N)
moyenne4 = sol[N-1]
bornesup4 = moyenne4 + 1.96/np.sqrt(N)*ecartype4
borneinf4 = moyenne4 - 1.96/np.sqrt(N)*ecartype4
diff4 = bornesup4 - borneinf4

plt.title("Fonction d'importance avec un alpha arbitraire simple (Schéma d'Euler)")
plt.grid(True)
plt.plot(np.linspace(1,N,N),sol)
plt.xlabel("Nombre de simulations")
plt.ylabel("Estimateur")
plt.axis([0,N,0.18,0.22])
plt.show()

print("Méthode de fonction d'importance avec un alpha arbitraire simple (Schéma d'Euler) :")
print("Estimateur :",moyenne4)
print("Ecart-type :",ecartype4)
print("Intervalle de confiance à 95% :",[borneinf4,bornesup4])
print("Erreur :",diff4/2)