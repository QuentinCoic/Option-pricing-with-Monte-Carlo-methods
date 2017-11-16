# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de fonction d'importance par gradient stochastique avec un alpha simple (Schéma d'Euler)

N=1000000
n=10
sigma=0.5
x0=0
K=1
T=1
delta=T/n
alphgradsimple=0#Valeur initiale de alpha 
sol=[0]
ecartype4grad=0

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)
    
def importance(x):#Quotient des deux densités
    return np.exp(-alphgradsimple*np.sum(x)+alphgradsimple**2/2)
    
def derivee(x):#Dérivée par rapport à alpha du quotient
    return (alphgradsimple-np.sum(x))*np.exp(-alphgradsimple*np.sum(x)+alphgradsimple**2/2)

for i in range (1,N+1):
    
    deltaW = np.random.randn(n)*np.sqrt(delta)#Simulation des incréments du mouvement brownien
    xdelta = x0
   
    for j in range (n):#Calcul de X_T_delta
            
        xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[j])
        
    xdeltaimp = xdelta + sigma*alphgradsimple#X_T_delta avec brownien translaté
    sol.append((i-1)/i*sol[i-1] + 1/i*g(xdeltaimp)*importance(deltaW + delta*alphgradsimple))#MAJ de l'estimateur
    ecartype4grad = (i-1)/i*ecartype4grad + 1/i*g(xdelta)**2*importance(deltaW)#MAJ de Y^2_N_barre
    alphgradsimple = alphgradsimple - 1/i*g(xdelta)**2*derivee(deltaW)#MAJ de alpha

moyenne4grad = sol[N-1]
ecartype4grad = np.sqrt(ecartype4grad - moyenne4grad**2)
bornesup4grad = moyenne4grad + 1.96/np.sqrt(N)*ecartype4grad
borneinf4grad = moyenne4grad - 1.96/np.sqrt(N)*ecartype4grad
diff4grad = bornesup4grad - borneinf4grad

plt.title("Gradient stochastique avec un alpha simple (Schéma d'Euler)")
plt.grid(True)
plt.plot(np.linspace(1,N,N),sol[1:])
plt.xlabel("Nombre de simulations")
plt.ylabel("Estimateur")
plt.axis([0,N,0.18,0.22])
plt.show()

print("Gradient stochastique avec un alpha simple (Schéma d'Euler) :")
print("alpha optimal :",alphgradsimple)
print("Estimateur :",moyenne4grad)
print("Ecart-type :",ecartype4grad)
print("Intervalle de confiance à 95% :",[borneinf4grad,bornesup4grad])
print("Erreur :",diff4grad/2)