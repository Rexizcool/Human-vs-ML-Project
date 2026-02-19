 import pandas as pd
import matplotlib as plt

penguin=pd.read_csv('/workspaces/Human-vs-ML-Project/data/penguins_size.csv')





adl_peng_lgth = penguin.loc[penguin['species'] == "Adelie", ['culmen_length_mm']]


