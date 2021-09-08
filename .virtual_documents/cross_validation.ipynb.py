import numpy as np
import pandas as pd
from sklearn import model_selection


df_train = pd.read_csv('input/30-days-of-ml/train.csv')


# kfold　column　追加
df_train['kfold'] = -1


# 訓練データを５つに分ける
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_indices, valid_indices) in enumerate(kf.split(X=df_train)):
    df_train.loc[valid_indices, 'kfold'] = fold


df_train.head()



