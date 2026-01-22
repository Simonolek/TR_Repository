import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import gurobipy

import oct_univariate_model
import oct_hyperplanes_model
import bus_dataset

solver = "gurobi"
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available()
SOLVER.options['TimeLimit'] = 120   # Stop after 600 seconds (10 mins)
SOLVER.options['MIPGap'] = 0.05     # Stop if within 5% of optimal

D = 3
Nmin = 5
data = bus_dataset.bus_data
labels = bus_dataset.bus_labels
alpha = 0.01

univariate_or_hyperplanes = "hyperplanes"

if univariate_or_hyperplanes == "univariate":

    bus_tree_model = oct_univariate_model.oct_univariate_model(D, Nmin, data, labels, alpha)
    bus_tree_model.write("mon_modele.lp", io_options={'symbolic_solver_labels': True})
    SOLVER.solve(bus_tree_model)

    print("Misclasifiation value:", pyo.value(bus_tree_model.misclasification_complexity))

    graph = oct_univariate_model.plot_oct_univariate(bus_tree_model)
    graph.view()
    

elif univariate_or_hyperplanes == "hyperplanes"  :

    bus_tree_model = oct_hyperplanes_model.oct_hyperplanes_model(D, Nmin, data, labels, alpha)
    bus_tree_model.write("mon_modele.lp", io_options={'symbolic_solver_labels': True})
    SOLVER.solve(bus_tree_model)

    print("Misclasifiation value:", pyo.value(bus_tree_model.misclasification_complexity))

    graph = oct_hyperplanes_model.plot_oct_hyperplanes(bus_tree_model)
    graph.view()

else :
    print("type of model not specified")


