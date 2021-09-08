import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# K-fold済みのデータ
df = pd.read_csv("../input/30days-folds/train_folds.csv")
df_test = pd.read_csv("../input/30-days-of-ml/test.csv")
sample_submission = pd.read_csv("../input/30-days-of-ml/sample_submission.csv")


useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
numerical_cols = [col for col in useful_features if col.startswith("cont")]
df_test = df_test[useful_features]


# only ordinal encoder

final_predictions = []
scores = []
for fold in range(5):
    xtrain =  df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])
        
    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
    
print(np.mean(scores), np.std(scores))


#standardlization

final_predictions = []
scores = []
for fold in range(5):
    # 
    xtrain =  df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])
    
    scaler = preprocessing.StandardScaler()
    xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])
    xvalid[numerical_cols] = scaler.transform(xvalid[numerical_cols])
    xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])
    
    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
    
print(np.mean(scores), np.std(scores))


#log transfomation

useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
numerical_cols = [col for col in useful_features if col.startswith("cont")]
df_test = df_test[useful_features]

for col in numerical_cols:
    df[col] = np.log1p(df[col])
    df_test[col] = np.log1p(df_test[col])


final_predictions = []
scores = []
for fold in range(5):
    xtrain =  df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])
    

    
    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
    
print(np.mean(scores), np.std(scores))


# polynomial features

useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
numerical_cols = [col for col in useful_features if col.startswith("cont")]
df_test = df_test[useful_features]

poly = preprocessing.PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
train_poly = poly.fit_transform(df[numerical_cols])
test_poly = poly.fit_transform(df_test[numerical_cols])

df_poly = pd.DataFrame(train_poly, columns=[f"poly_{i}" for i in range(train_poly.shape[1])])
df_test_poly = pd.DataFrame(test_poly, columns=[f"poly_{i}" for i in range(test_poly.shape[1])])

df = pd.concat([df, df_poly], axis=1)
df_test = pd.concat([df_test, df_test_poly], axis=1)

useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
numerical_cols = [col for col in useful_features if col.startswith("cont")]
df_test = df_test[useful_features]

final_predictions = []
scores = []
for fold in range(5):
    xtrain =  df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])
    

    
    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
    
print(np.mean(scores), np.std(scores))


# one hot encoding

useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
numerical_cols = [col for col in useful_features if col.startswith("cont")]
df_test = df_test[useful_features]

for col in numerical_cols:
    df[col] = np.log1p(df[col])
    df_test[col] = np.log1p(df_test[col])


final_predictions = []
scores = []
for fold in range(5):
    xtrain =  df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    xtrain_ohe = ohe.fit_transform(xtrain[object_cols])
    xvalid_ohe = ohe.transform(xvalid[object_cols])
    xtest_ohe = ohe.transform(xtest[object_cols])
    
    xtrain_ohe = pd.DataFrame(train_poly, columns=[f"ohe_{i}" for i in range(xtrain_ohe.shape[1])])
    xtrain_ohe = pd.DataFrame(train_poly, columns=[f"ohe_{i}" for i in range(xtrain_ohe.shape[1])])
    xtrain_ohe = pd.DataFrame(train_poly, columns=[f"ohe_{i}" for i in range(xtrain_ohe.shape[1])])
    
    xtrain = pd.DataFrame([x_train, xtrain_ohe], axis=1)
    xvalid = pd.DataFrame([x_valid, xvalid_ohe], axis=1)
    xtest = pd.DataFrame([xtest, xtest_ohe], axis=1)
    

    
    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
    
print(np.mean(scores), np.std(scores))


# one hot encoding of categorical valiables + standarization of ohe & numerical
