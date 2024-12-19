from turtle import pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

train_proteins = pd.read_csv("./train_proteins.csv")
train_peptides = pd.read_csv("./train_peptides.csv")
train_1 = pd.read_csv("./train_1.csv")
train_2 = pd.read_csv("./train_2.csv")
train_3 = pd.read_csv("./train_3.csv")

model = {i: {} for i in range(4)}
mms = MinMaxScaler()
n_estimators = [5, 20, 50, 100]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 120, num=12)]
min_samples_split = [2, 6, 10]
min_samples_leaf = [1, 3, 4]
bootstrap = [True, False]

# SMAPE function
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)
# Lists to store metrics for each model
smape_scores_model = [[] for _ in range(3)]
cv_scores_model = [[] for _ in range(3)]
for i in range(3):
    current = globals()['train_{0}'.format(i+1)]  # Corrected loop variable

    print('--------------------------------------------------------')
    print('Model {0}'.format(i + 1))

    X_train, X_test, y_train, y_test = train_test_split(
        current.drop(columns=['patient_id', 'updrs_{0}'.format(i + 1)], axis=1),
        current['updrs_{0}'.format(i + 1)].astype(np.float32),
        test_size=0.2, random_state=42
    )
    model[i] = {}  # Initialize the dictionary for model i

    # Random Forest Regressor
    rfc = RandomForestRegressor()
    forest_params = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}

    print('Fitting Random Forest...')
    try:
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        rf_cv_scores = np.mean(cross_val_score(rfc, X_train, y_train, cv=cv, scoring=make_scorer(smape)))
        print('Random Forest Average Cross-Validation Score:', rf_cv_scores)
        
        clf_rf = RandomizedSearchCV(rfc, forest_params, cv=cv, scoring=make_scorer(smape), verbose=0)
        clf_rf.fit(X_train, y_train)
    except Exception as e:
        print('Error fitting Random Forest:', e)

    if hasattr(clf_rf, 'best_params_'):
        print('Random Forest Best Params:', clf_rf.best_params_)
        rf_smape=  smape(y_test, clf_rf.predict(X_test))
        print('Random Forest Test SMAPE:',rf_smape)
    else:
        print('Random Forest fitting failed. Check data and hyperparameters.')

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()
    gbr_params = {'n_estimators': [50, 100, 200],
                  'learning_rate': [0.01, 0.1, 0.2],
                  'max_depth': [3, 5, 7]}

    print('Fitting Gradient Boosting...')
    try:
        gbr_cv_scores = np.mean(cross_val_score(gbr, X_train, y_train, cv=cv, scoring=make_scorer(smape)))
        print('Gradient Boosting Average Cross-Validation Score:', gbr_cv_scores)
        
        clf_gbr = RandomizedSearchCV(gbr, gbr_params, cv=cv, scoring=make_scorer(smape), verbose=0)
        clf_gbr.fit(X_train, y_train)
    except Exception as e:
        print('Error fitting Gradient Boosting:', e)

    if hasattr(clf_gbr, 'best_params_'):
        print('Gradient Boosting Best Params:', clf_gbr.best_params_)
        gr_smape= smape(y_test, clf_gbr.predict(X_test))
        print('Gradient Boosting Test SMAPE:', )
    else:
        print('Gradient Boosting fitting failed. Check data and hyperparameters.')

    # CART (Decision Tree Regressor)
    cart = DecisionTreeRegressor()
    cart_params = {'max_depth': [None, 10, 20, 30],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4]}

    print('Fitting CART...')
    try:
        cart_cv_scores = np.mean(cross_val_score(cart, X_train, y_train, cv=cv, scoring=make_scorer(smape)))
        print('CART Average Cross-Validation Score:', np.mean(cart_cv_scores))
        
        clf_cart = RandomizedSearchCV(cart, cart_params, cv=cv, scoring=make_scorer(smape), verbose=0)
        clf_cart.fit(X_train, y_train)
    except Exception as e:
        print('Error fitting CART:', e)

    if hasattr(clf_cart, 'best_params_'):
        print('CART Best Params:', clf_cart.best_params_)
        cart_smape= smape(y_test, clf_cart.predict(X_test))
        print('CART Test SMAPE:',cart_smape)
    else:
        print('CART fitting failed. Check data and hyperparameters.')

    model[i] = {'RandomForest': clf_rf, 'GradientBoosting': clf_gbr, 'CART': clf_cart,}
    smape_scores_model[i].extend([cart_smape, gr_smape, rf_smape])
    cv_scores_model[i].extend([cart_cv_scores, gbr_cv_scores, rf_cv_scores])
    

for i in range(3):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    # SMAPE Scores
    axes[0].barh(['Random Forest', 'Gradient Boosting', 'CART'], smape_scores_model[i], color='blue')
    axes[0].set_title(f'Model {i + 1} - SMAPE Scores')
    axes[0].set_xlabel('SMAPE')


    # Cross-Validation Scores
    axes[1].barh(['Random Forest', 'Gradient Boosting', 'CART'], cv_scores_model[i], color='orange')
    axes[1].set_title(f'Model {i + 1} - Cross-Validation Scores')
    axes[1].set_xlabel('Cross-Validation Score')

    plt.tight_layout()
    plt.show()

