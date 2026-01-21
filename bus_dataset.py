import random
import pandas as pd
import numpy as np

np.random.seed(10)

# Construire  le bus_dataset bus

n = 100
D = np.random.uniform(low=0.0, high=50.0, size=n)
k1 = np.random.uniform(low=0.0, high=20.0, size=n)
bus_data = np.vstack((D,k1)).transpose()

bus_labels = []
for i in range(n):
    if bus_data[i,1] <= 5:
        if bus_data[i,0] - 2*bus_data[i,1] < 30:
            bus_labels.append("A")
            continue
        else : 
            bus_labels.append("B")
            continue
    else:
        if bus_data[i,0] + bus_data[i,1] < 60:
            bus_labels.append("A")
            continue
        else :
            if bus_data[i,1] < 12.5 :
                bus_labels.append("A")
                continue
            else :
                bus_labels.append("C")
                continue

bus_data = pd.DataFrame(bus_data, columns = ["D", "k1"])
bus_labels = np.array(bus_labels)
print(bus_labels)








        



        

