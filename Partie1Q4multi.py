# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

#Méthode de fonction d'importance avec un alpha arbitraire multiple (Schéma d'Euler)

N=1000000
n=10
sigma=0.5
x0=0
K=1
T=1
delta=T/n
alpharbmulti=np.array([0.31,0.44,0.43,0.28,0.31,0.33,0.27,0.43,0.18,0.43])#Valeur de alpha choisie arbitrairement
sol=[]

def u(x):
    return 0.1*(np.sqrt(np.exp(x))-1)-1/8

def g(x):
    return np.maximum(np.exp(x)-K,0)
    
def importance(x):#Quotient des deux densités
    return np.exp(-np.sum(x*alpharbmulti)+delta/2*np.sum(alpharbmulti*alpharbmulti))
    
for i in range (N):
    
    deltaW = []
    
    for l in range(n):
        
        deltaW = np.random.randn(n)*np.sqrt(delta) + delta*alpharbmulti#Simulation des incréments du mouvement brownien translatés de alpha*delta
    
    xdelta = x0
   
    for j in range (n):#Calcul de X_T_delta avec brownien translaté
            
        xdelta = xdelta + u(xdelta)*delta + sigma*(deltaW[j])
        
    sol.append(g(xdelta)*importance(deltaW))

ecartype4mult = np.std(sol)
sol = np.cumsum(sol)/np.linspace(1,N,N)
moyenne4mult = sol[N-1]
bornesup4mult = moyenne4mult + 1.96/np.sqrt(N)*ecartype4mult
borneinf4mult = moyenne4mult - 1.96/np.sqrt(N)*ecartype4mult
diff4mult = bornesup4mult - borneinf4mult

plt.title("Fonction d'importance avec un alpha arbitraire multiple (Schéma d'Euler)")
plt.grid(True)
plt.plot(np.linspace(1,N,N),sol)
plt.xlabel("Nombre de simulations")
plt.ylabel("Estimateur")
plt.axis([0,N,0.18,0.22])
plt.show()

print("Méthode de fonction d'importance avec un alpha arbitraire multiple (Schéma d'Euler) :")
print("Estimateur :",moyenne4mult)
print("Ecart-type :",ecartype4mult)
print("Intervalle de confiance à 95% :",[borneinf4mult,bornesup4mult])
print("Erreur :",diff4mult/2)