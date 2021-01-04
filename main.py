# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams

register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
rcParams['figure.figsize'] = 22, 10

# %%
df = pd.read_csv("./datasets/london_merged.csv",
                 parse_dates=['timestamp'], index_col='timestamp')
df.head()
# %%
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month
df.head()
# %%
sns.lineplot(x=df.index, y='cnt', data=df)
# %%
df_month = df.resample('M').sum()
sns.lineplot(x=df_month.index, y='cnt', data=df_month)
# %%
sns.pointplot(x='hour', y='cnt', data=df)
# %%
sns.pointplot(x='hour', y='cnt', data=df, hue='is_holiday')
# %%
sns.pointplot(x='day_of_week', y='cnt', data=df)
# %%
