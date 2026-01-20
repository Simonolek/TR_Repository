from imodels.util.data_util import get_clean_dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X, y, feature_names = get_clean_dataset("compas_two_year_clean")

print(len(feature_names))

X = pd.DataFrame(X, columns = feature_names)
X = X.drop(columns=["age_cat:25_-_45", "age_cat:Greater_than_45", "age_cat:Less_than_25" ])
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaled_array, columns = X.columns)

y = pd.DataFrame(y)


print(sorted(list(set(X["c_jail_time"].values))))



