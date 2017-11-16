# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de fonction d'importance par gradient stochastique avec un alpha multiple (Schéma d'Euler)

N=1000000
n=10
sigma=0.5
x0=0
K=1
T=1
delta=T/n
alphgradmulti=np.zeros(n)#Valeur initiale de alpha  
sol=[0]
ecartype4multi=0

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)
    
def importance(x):#Quotient des deux densités
    return np.exp(-np.sum(x*alphgradmulti)+delta/2*np.sum(alphgradmulti*alphgradmulti))
    
def derivee(x):#Gradient par rapport à alpha du quotient
    return (delta*alphgradmulti-x)*np.exp(-np.sum(x*alphgradmulti)+delta/2*np.sum(alphgradmulti*alphgradmulti))

for i in range (1,N+1):
    
    deltaW = np.random.randn(n)*np.sqrt(delta)#Simulation des incréments du mouvement brownien
    xdelta = x0
   
    for j in range (n):#Calcul de X_T_delta
            
        xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[j])
        
    xdeltaimp = xdelta + sigma*delta*np.sum(alphgradmulti)#X_T_delta avec brownien translaté
    sol.append((i-1)/i*sol[i-1] + 1/i*g(xdeltaimp)*importance(deltaW + delta*alphgradmulti))#MAJ de l'estimateur
    ecartype4multi = (i-1)/i*ecartype4multi + 1/i*g(xdelta)**2*importance(deltaW)#MAJ de Y^2_N_barre
    alphgradmulti = alphgradmulti - 1/i*g(xdelta)**2*derivee(deltaW)#MAJ de alpha

moyenne4multi = sol[N-1]
ecartype4multi = np.sqrt(ecartype4multi - moyenne4multi**2)
bornesup4multi = moyenne4multi + 1.96/np.sqrt(N)*ecartype4multi
borneinf4multi = moyenne4multi - 1.96/np.sqrt(N)*ecartype4multi
diff4multi = bornesup4multi - borneinf4multi

plt.title("Gradient stochastique avec un alpha multiple (Schéma d'Euler)")
plt.grid(True)
plt.plot(np.linspace(1,N,N),sol[1:])
plt.xlabel("Nombre de simulations")
plt.ylabel("Estimateur")
plt.axis([0,N,0.18,0.22])
plt.show()

print("Gradient stochastique avec un alpha multiple (Schéma d'Euler) :")
print("alpha optimal :",alphgradmulti)
print("Estimateur :",moyenne4multi)
print("Ecart-type :",ecartype4multi)
print("Intervalle de confiance à 95% :",[borneinf4multi,bornesup4multi])
print("Erreur :",diff4multi/2)