# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Activation, Dropout
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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
train_size = int(len(df)*0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
print(train.shape, test.shape)
# %%
f_columns = ['t1', 't2', 'hum', 'wind_speed']
f_transformer = RobustScaler()
cnt_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['cnt']].to_numpy())
# %%
train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['cnt'] = cnt_transformer.transform(train[['cnt']].to_numpy())
# %%
test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['cnt'] = cnt_transformer.transform(test[['cnt']].to_numpy())
# %%
train.head()
# %%


def create_dataset(X, Y, time_steps=1):
    Xs, Ys = list(), list()
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].to_numpy())
        Ys.append(Y.iloc[i+time_steps])
    return np.array(Xs), np.array(Ys)


# %%
TIME_STEPS = 24
X_train, Y_train = create_dataset(train, train.cnt, TIME_STEPS)
X_test, Y_test = create_dataset(test, test.cnt, TIME_STEPS)
print(X_train.shape, Y_train.shape)
# %%
bilstm_model = Sequential([
    Bidirectional(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.2),
    Dense(1)
])

# %%
bilstm_model.compile(optimizer='adam', loss='mean_squared_error')
history = bilstm_model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_split=0.2,
                           shuffle=False, verbose=2)
# %%
