import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


get_ipython().getoutput("pip install xgboost")


train = pd.read_csv('input/30-days-of-ml/train_folds.csv')
test = pd.read_csv('input/30-days-of-ml/test.csv')
sample_submission = pd.read_csv('input/30-days-of-ml/sample_submission.csv')


train.head()


test.head()


sample_submission.head()


train.columns


y = train.target
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

train.head()


train.head()
train.drop('kFold', axis=1)


train.drop(['kFold'], axis=1, inplace=True)
train



test


print("Train shape: ", train.shape, "\nTest shape: ", test.shape)


from sklearn.preprocessing import OrdinalEncoder
cat_cols = [col for col in train.columns if 'cat' in col]

X = train.copy()
X_test = test.copy()
enc = OrdinalEncoder()
X[cat_cols] = enc.fit_transform(train[cat_cols])
X_test[cat_cols] = enc.transform(test[cat_cols])
X.head()


X_test.head()


X.head()



X_stats = X.describe()
X_stats = X_stats.transpose()
X_stats


#正規化

def norm(x):
    return (x - X_stats['mean']) / X_stats['std']

normed_X = norm(X)
normed_X_test = norm(X_test)


normed_X.head()


from tqdm import tqdm
import gc
from functools import reduce
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras


import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

tf.random.set_seed(777)
NFOLDS = 6
skf = StratifiedKFold(n_splits=NFOLDS)
# folds = skf.split(X, cl)
# folds = list(folds)


def def_model():
#     inp = L.Input(name="inputs", shape=(n_in,))
    model = keras.Sequential([
        L.Dense(64, activation="relu", name="d1", input_shape=[len(X.keys())]), 
        L.Dense(64, activation="relu", name="d2") 
        L.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
#     preds = L.Dense(4, activation="linear", name="preds")(x)

    model.compile(loss="mse", optimizer="adam",
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model

net = def_model()
print(net.summary())

# oof = np.zeros(y.shape)
# nets = []
# for idx in range(NFOLDS):
#     print("FOLD:", idx)
#     tr_idx, val_idx = folds[idx]
#     ckpt = ModelCheckpoint(f"w{idx}.h5", monitor='val_loss', verbose=1, save_best_only=True,mode='min')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.0005)
#     es = EarlyStopping(monitor='val_loss', patience=6)
#     reg = make_model(X.shape[1])
# #     reg.fit(X[tr_idx], y[tr_idx], epochs=10, batch_size=35_000, validation_data=(X[val_idx], y[val_idx]),
# #             verbose=1, callbacks=[ckpt, reduce_lr, es])
#     reg.load_weights(f"w{idx}.h5")
#     oof[val_idx] = reg.predict(X[val_idx], batch_size=50_000, verbose=1)
#     nets.append(reg)
#     gc.collect()
#     #
# #

# mae = mean_absolute_error(y, oof)
# rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_id]))
# print("mae:", mae)
# print("rmse:", rmse)


# del tr



example_batch = normed_X[:10]
example_result = net.predict(example_batch)
example_result


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
EPOCHS = 1000

history = net.fit(
normed_X, y,
epochs=EPOCHS, validation_split=0.2, verbose=0,
callbacks=[PrintDot()])



