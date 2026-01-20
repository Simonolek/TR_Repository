import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import highspy

import oct_model
import bus_dataset

solver = "appsi_highs"
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available()

D = 3
Nmin = 0
data = bus_dataset.bus_data
labels = bus_dataset.bus_labels
alpha = 0

bus_tree_model = oct_model.oct_model(D, Nmin, data, labels, alpha)
bus_tree_model.write("mon_modele.lp", io_options={'symbolic_solver_labels': True})
SOLVER.solve(bus_tree_model)

print("Misclasifiation value:", pyo.value(bus_tree_model.misclasification_complexity))

graph = oct_model.plot_oct(bus_tree_model)
graph.view()


