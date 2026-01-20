import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
import graphviz


# data set is expected to be normalized to [0,1]
# data is expected to be a pd.DataFrame, labels is expected to be a list or an array

def oct_model(D, Nmin, data, labels, alpha): # D, labels = {1,...,K} les K labels différents
    
    
    features = np.array(data.columns)

    # Making data and labels array 

    data = np.array(data)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    # Defining n and p

    n, p = data.shape
    K = len(np.unique(labels))

    # Mapping labels to {1, ..., K}

    mapping_labels = {}
    for l in range(len(unique_labels)):
        mapping_labels[l+1] = unique_labels[l]

    # Mapping features to {1, ..., p}

    mapping_features = {}
    for l in range(len(features)):
        mapping_features[l+1] = features[l]



    # ^L (baseline accuracy)

    values, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    L_hat = n - max_count
    if L_hat == 0 : L_hat = 1.0

    # T (number of nodes)

    T = 2**(D+1)-1

    # Espilon

    eps = {}
    for j in range(1, p+1):
        col = data[:, j-1]
        unique_sorted = sorted(list(set(col)))
        if len(unique_sorted) < 2 :
            eps[j] = 0.001
            continue
        diffs = [unique_sorted[i+1] - unique_sorted[i] for i in range(len(unique_sorted)-1)]
        eps[j] = min(diffs)
    
    epsmax = max(eps.values())
        

    # Y "matrix"

    Y={}
    for i in range(1, n+1):
        for k in range(1, K+1):
            Y[i,k] = 1 if labels[i-1] == mapping_labels[k] else 0

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
    m.K = pyo.Param(initialize = K) # nombre de labels
    m.Nmin = pyo.Param(initialize = Nmin) # nombre de data minimum dans un leaf
    m.alpha = pyo.Param(initialize = alpha)


    # Sets
    
    m.NODES = pyo.RangeSet((m.T)) # index of all nodes
    m.BRANCHES = pyo.RangeSet(1, pyo.value(m.T)//2) # index of branch nodes
    m.LEAVES = pyo.RangeSet(pyo.value(m.T)//2 + 1, m.T) # index of leaf nodes
    m.FEATURES = pyo.RangeSet(m.p)
    m.DATA = pyo.RangeSet(m.n)
    m.LABELS = pyo.RangeSet(m.K)

    # Params

    m.X = pyo.Param(m.DATA, m.FEATURES, initialize=lambda m, i, j: data[i-1][j-1])
    m.Y = pyo.Param(m.DATA, within=pyo.Any, initialize = lambda m, i : labels[i-1])

    # Variables
   
    m.a = pyo.Var(m.FEATURES, m.BRANCHES, domain = pyo.Binary) # aTx < b, a étant un p-vecteur où un seul élement vaut 1 et le reste 0
    m.b = pyo.Var(m.BRANCHES, domain = pyo.NonNegativeReals, bounds = (0, 1))
    m.d = pyo.Var(m.BRANCHES, domain = pyo.Binary) # d vaut 1 si il y a un split au noeud t, 0 sinon
    m.z = pyo.Var(m.DATA, m.LEAVES, domain = pyo.Binary) # zit vaut 1 si xi is in node t
    m.l = pyo.Var(m.LEAVES, domain = pyo.Binary) #lt vaut 1 si le leaf t contient au moins 1 point
    m.Nkt = pyo.Var(m.LABELS, m.LEAVES, domain = pyo.NonNegativeReals, bounds=(0, m.n))
    m.Nt = pyo.Var(m.LEAVES, domain = pyo.NonNegativeReals, bounds=(0, m.n))
    m.c = pyo.Var(m.LABELS, m.LEAVES, domain = pyo.Binary)
    m.L = pyo.Var(m.LEAVES, domain = pyo.NonNegativeReals, bounds=(0, m.n))

    # Mappings pour la fonction plot

    m.mapping_labels = mapping_labels
    m.mapping_features = mapping_features

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
    
    @m.Constraint(m.DATA, m.LEAVES, m.BRANCHES)
    def prevent_left_if_no_split(m, i, t, ml):
        if ml in leftanc[t]:
             return m.z[i,t] <= m.d[ml]
        return pyo.Constraint.Skip
    
    @m.Constraint(m.LABELS, m.LEAVES)
    def number_points_label_k_in_leaf_t(m, k, t):
        return m.Nkt[k,t] == pyo.quicksum(Y[i,k]*m.z[i,t] for i in m.DATA) 

    @m.Constraint(m.LEAVES)
    def number_points_in_leaf_t(m, t):
        return m.Nt[t] == pyo.quicksum(m.z[i,t] for i in m.DATA) 
    
    @m.Constraint(m.LEAVES)
    def single_class_prediction(m, t):
        return pyo.quicksum(m.c[k,t] for k in m.LABELS) == m.l[t]
    
    @m.Constraint(m.LABELS, m.LEAVES)
    def L_definition_1(m, k, t):
        return m.L[t] >= m.Nt[t] - m.Nkt[k,t] - m.n*(1-m.c[k,t])

    @m.Constraint(m.LABELS, m.LEAVES)
    def L_definition_2(m, k, t):
        return m.L[t] <= m.Nt[t] - m.Nkt[k,t] + m.n*m.c[k,t]

    # Objective function

    @m.Objective(sense = pyo.minimize)
    def misclasification_complexity(m):
        return (1/L_hat) * pyo.quicksum(m.L[t] for t in m.LEAVES) + m.alpha*pyo.quicksum(m.d[t] for t in m.BRANCHES)
        
    return m






def plot_oct(m):

    dot = graphviz.Digraph(comment = "OCT")

    mapping_labels = getattr(m, "mapping_labels", {})
    mapping_features = getattr(m, "mapping_features", {})

    for t in m.NODES:
        node_id = str(t) 
        if t in m.BRANCHES:
            d_val = pyo.value(m.d[t])
            if d_val > 0.5:
                for j in m.FEATURES:
                    active_str = "error : no active"
                    if pyo.value(m.a[j,t]) > 0.5 :
                        active = mapping_features[j]
                        active_str = str(active)
                        break
                b_str = str(pyo.value(m.b[t]))
                label = active_str + "<" + b_str
                dot.node(node_id, label, shape = "oval", style = "filled", fillcolor = "white")
                dot.edge(node_id, str(2*t), label = "True")
                dot.edge(node_id, str(2*t+1), label = "False")
            else :
                dot.node(node_id, "Pass", shape='oval', style='dotted', fontcolor='gray', color='gray')
                dot.edge(node_id, str(2*t+1), style = "dashed", color = "gray")
        else:
            l_val = pyo.value(m.l[t])
            if l_val > 0.5 :
                for k in m.LABELS:
                    if pyo.value(m.c[k,t]) > 0.5:
                        label = mapping_labels[k]
                        label_str = str(label)
                        break
                dot.node(node_id, label = label_str)
        
    return dot







