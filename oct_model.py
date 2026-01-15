import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

def oct_model(D):
    
    # Model
    m = pyo.ConcreteModel("OCT")

    # Params
    m.T = pyo.Param(initialize = 2**(D+1)-1)

    # Sets
    m.TSET = pyo.RangeSet(1, (m.T)//2)
    m.TBSET = pyo.RangeSet(1, (m.T)//2)
    m.TLSET = pyo.RangeSet((m.T)//2 + 1, m.T)
    return None
