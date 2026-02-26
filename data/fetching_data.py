from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_penguin_data():
    # ---------------------------------------------------------------------
    # Copied precisely from the UCI Machine Learning Repository package documentation

    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data imported as pandas dataframes
    X = penguin.data.features
    y = penguin.data.targets
    
    # ---------------------------------------------------------------------

    feature_names = iris.variables[iris.variables['role'] == 'Feature']['name'].tolist()
    target_name = iris.variables[iris.variables['role'] == 'Target']['name'].values[0]

    df = pd.DataFrame(iris.data.features, columns=feature_names)
    df[target_name] = iris.data.targets

    # I printed here to confirm that I was importing data correctly.
    # print(df.head())

    return df, target_name