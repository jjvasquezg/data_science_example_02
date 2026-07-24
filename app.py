## Importe de librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

## Carga de datos

df = pd.read_csv('datasets/agro_colombia.csv')

df.info()