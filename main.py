import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import highspy

import oct_univariate_model
import oct_hyperplanes_model
import bus_dataset

solver = "appsi_highs"
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available()

D = 2
Nmin = 0
data = bus_dataset.bus_data
labels = bus_dataset.bus_labels
alpha = 0

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


