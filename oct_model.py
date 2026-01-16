import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


# data set is expected to be normalized to [0,1]

def oct_model(D, n, p, Nmin, data, labels): # D, n = le nombre de data, p = le nombre de features, labels = {1,...,K} les K labels différents
    
    # T (number of nodes)

    T = 2**(D+1)-1

    # Espilon
    eps = {}
    for j in range(1, p+1):
        col = [row[j-1] for row in data]
        unique_sorted = sorted(list(set(col)))
        if len(unique_sorted) < 2 :
            eps[j] = 0.001
            continue
        diffs = [unique_sorted[i+1] - unique_sorted[i] for i in range(len(unique_sorted)-1)]
        eps[j] = min(diffs)
    epsmax = max(eps.values())


    # Ancestors dictionnaries

    anc= {}
    anc[1] = []
    for i in range(2, T + 1):
        anc[i] = [i//2] + anc[i//2]

    leftanc = {}
    rightanc = {}
    leftanc[1] = []
    rightanc[1] = []
    for i in range(2, T + 1):
        if i%2 == 0:
            leftanc[i] = [i//2] + leftanc[i//2]
            rightanc[i] = rightanc[i//2] 
        else :
            leftanc[i] = leftanc[i//2]
            rightanc[i] = [i//2] + rightanc[i//2] 

    # Model
    
    m = pyo.ConcreteModel("OCT")

    # Params
    
    m.D = pyo.Param(initialize = D) # depht de l'arbre
    m.T = pyo.Param(initialize = T) # number of nodes in the tree 
    m.p = pyo.Param(initialize = p) # nommbre de features
    m.n = pyo.Param(initialize = n) # nombre de données
    m.Nmin = pyo.Param(initialize = Nmin) # nombre de data minimum dans un leaf


    # Sets
    
    m.NODES = pyo.RangeSet((m.T)) # index of all nodes
    m.BRANCHES = pyo.RangeSet(1, (m.T)//2) # index of branch nodes
    m.LEAVES = pyo.RangeSet((m.T)//2 + 1, m.T) # index of leaf nodes
    m.FEATURES = pyo.RangeSet(m.p)
    m.DATA = pyo.RangeSet(m.n)

    # Params

    m.X = pyo.Param(m.DATA, m.FEATURES, initialize=lambda m, i, j: data[i-1][j-1])
    m.Y = pyo.Param(m.DATA, initialize = lambda m, i : labels[i-1])

    # Variables
   
    m.a = pyo.Var(m.FEATURES, m.BRANCHES, domain = pyo.Binary) # aTx < b, a étant un p-vecteur où un seul élement vaut 1 et le reste 0
    m.b = pyo.Var(m.BRANCHES, domain = pyo.NonNegativeReals)
    m.d = pyo.Var(m.BRANCHES, domain = pyo.Binary) # d vaut 1 si il y a un split au noeud t, 0 sinon
    m.z = pyo.Var(m.DATA, m.LEAVES, domain = pyo.Binary) # zit vaut 1 si xi is in node t
    m.l = pyo.Var(m.LEAVES, domain = pyo.Binary) #lt vaut 1 si le leaf t contient au moins 1 point

    # Constraints

    @m.Constraint(m.BRANCHES)
    def split_must_exist_a(m, t):
        return pyo.quicksum(m.a[j, t] for j in m.FEATURES) == m.d[t]
    
    @m.Constraint(m.BRANCHES)
    def split_must_exist_b(m, t):
        return m.b[t] <= m.d[t]
    
    @m.Constraint(m.BRANCHES)
    def active_parent(m,t):
        if t==1:
            return pyo.Constraint.Skip
        else :
            return m.d[t] <= m.d[anc[t][0]]
        
    @m.Constraint(m.DATA, m.LEAVES)
    def cannot_be_in_empty_leaf(m, i, t):
        return m.z[i,t] <= m.l[t]

    @m.Constraint(m.LEAVES)
    def must_containt_Nmin(m, t):
        return pyo.quicksum(m.z[i,t] for i in m.DATA) >= m.Nmin*m.l[t]
    
    @m.Constraint(m.DATA)
    def cannot_be_in_several_leaves(m, i):
        return pyo.quicksum(m.z[i,t] for t in m.LEAVES) == 1
    
    @m.Constraint(m.DATA, m.LEAVES, m.BRANCHES)
    def left_branch_split(m, i, t, ml):
        if ml not in leftanc[t]:
            return pyo.Constraint.Skip
        return pyo.quicksum(m.a[j,ml]*(m.X[i,j] + eps[j]) for j in m.FEATURES) <= m.b[ml] + (1+epsmax)*(1-m.z[i,t])
        
    @m.Constraint(m.DATA, m.LEAVES, m.BRANCHES)
    def right_branch_split(m, i, t, mr):
        if mr not in rightanc[t]:
            return pyo.Constraint.Skip
        return pyo.quicksum(m.a[j,mr]*m.X[i,j] for j in m.FEATURES) >= m.b[mr] - (1-m.z[i,t])
        
    return m
