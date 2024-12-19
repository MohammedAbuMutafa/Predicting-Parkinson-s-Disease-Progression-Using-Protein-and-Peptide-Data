from turtle import pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold, train_test_split, cross_val_score, RandomizedSearchCV


import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


train_1 = pd.read_csv("./train_1.csv")
train_2 = pd.read_csv("./train_2.csv")
train_3 = pd.read_csv("./train_3.csv")

model = {i: {} for i in range(4)}
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
accuracy_scores_model = [[] for _ in range(3)]
cv_scores_model = [[] for _ in range(3)]

for i in range(3):
    current = globals()['train_{0}'.format(i + 1)]  # Corrected loop variable

    print('--------------------------------------------------------')
    print('Model {0}'.format(i + 1))

    X_train, X_test, y_train, y_test = train_test_split(
        current.drop(columns=['patient_id', 'updrs_{0}'.format(i + 1)], axis=1),
        current['updrs_{0}'.format(i + 1)].astype(np.int),  
        test_size=0.2, random_state=42
    )
    model[i] = {}  # Initialize the dictionary for model i

    # Random Forest Classifier
    rfc_classifier = RandomForestClassifier()
    try:
        cv_classifier = KFold(n_splits=10, shuffle=True, random_state=42)
        clf_rf_classifier = RandomizedSearchCV(rfc_classifier, {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }, cv=cv_classifier, scoring='f1_macro', verbose=0)

        clf_rf_classifier.fit(X_train, y_train)

        # Calculate SMAPE for the testing set
        smape_test = smape(y_test, clf_rf_classifier.predict(X_test))
        print('Random Forest Classifier Test SMAPE:', smape_test)

        # Add F1 score to the metrics dictionary
        f1 = f1_score(y_test, clf_rf_classifier.predict(X_test), average='micro')
        print('Random Forest Classifier Test F1 Score:', f1)

        # Cross-validation score
        cv_score = np.mean(cross_val_score(clf_rf_classifier.best_estimator_, X_train, y_train, cv=cv_classifier,
                                           scoring='f1_macro'))
        print('Random Forest Classifier Cross-Validation Score:', cv_score)

        model[i]['RandomForestClassifier'] = clf_rf_classifier



    except Exception as e:
        print('Error fitting Random Forest Classifier:', e)

    # Gradient Boosting
    gradient_boosting_classifier = GradientBoostingClassifier()
    try:
        clf_gradient_boosting = RandomizedSearchCV(gradient_boosting_classifier, {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0]
        }, cv=cv_classifier, scoring='f1_macro', verbose=0)

        clf_gradient_boosting.fit(X_train, y_train)

        # Calculate SMAPE for testing set
        smape_test_gb = smape(y_test, clf_gradient_boosting.predict(X_test))
        print('Gradient Boosting Test SMAPE:', smape_test_gb)

        # Add F1 score to the metrics dictionary
        f1_gb = f1_score(y_test, clf_gradient_boosting.predict(X_test), average='micro')
        print('Gradient Boosting Test F1 Score:', f1_gb)

        # Cross-validation score
        cv_score_gb = np.mean(cross_val_score(clf_gradient_boosting.best_estimator_, X_train, y_train, cv=cv_classifier,
                                           scoring='f1_macro'))
        
        print('Gradient Boosting Cross-Validation Score:', cv_score_gb)
        model[i]['GradientBoosting'] = clf_gradient_boosting



    except Exception as e:
        print('Error fitting Gradient Boosting:', e)

    # Decision Tree
    decision_tree_classifier = DecisionTreeClassifier()
    try:
        clf_decision_tree = RandomizedSearchCV(decision_tree_classifier, {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }, cv=cv_classifier, scoring='f1_macro', verbose=0)

        clf_decision_tree.fit(X_train, y_train)

        # Calculate SMAPE for testing set
        smape_test_dt = smape(y_test, clf_decision_tree.predict(X_test))
        print('Decision Tree Test SMAPE:', smape_test_dt)

        # Add F1 score to the metrics dictionary
        f1_dt = f1_score(y_test, clf_decision_tree.predict(X_test), average='micro')
        print('Decision Tree Test F1 Score:', f1_dt)

        # Cross-validation score
        cv_score_dt = np.mean(cross_val_score(clf_decision_tree.best_estimator_, X_train, y_train, cv=cv_classifier,
                                           scoring='f1_macro'))
        
        print('Decision Tree Cross-Validation Score:', cv_score_dt)
        
        model[i]['DecisionTree'] = clf_decision_tree


    except Exception as e:
        print('Error fitting Decision Tree:', e)

    mlp_classifier = MLPClassifier()
    try:
        clf_mlp = RandomizedSearchCV(mlp_classifier, {
            'hidden_layer_sizes': [(100,), (50, 50), (30, 20, 10)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }, cv=cv_classifier, scoring='f1_macro', verbose=0)

        clf_mlp.fit(X_train, y_train)

        # Calculate SMAPE for testing set
        smape_test_mlp = smape(y_test, clf_mlp.predict(X_test))
        print('MLP Classifier Test SMAPE:', smape_test_mlp)

        # Add F1 score to the metrics dictionary
        f1_mlp = f1_score(y_test, clf_mlp.predict(X_test), average='micro')
        print('MLP Classifier Test F1 Score:', f1_mlp)

        # Cross-validation score
        cv_score_mlp = np.mean(cross_val_score(clf_mlp.best_estimator_, X_train, y_train, cv=cv_classifier,
                                           scoring='f1_macro'))
        
        print('Decision Tree Cross-Validation Score:', cv_score_mlp)

        model[i]['MLPClassifier'] = clf_mlp



    except Exception as e:
        print('Error fitting MLP Classifier:', e)
      # Append scores to the lists for each model
    smape_scores_model[i].extend([smape_test, smape_test_gb, smape_test_dt, smape_test_mlp])
    accuracy_scores_model[i].extend([f1, f1_gb, f1_dt, f1_mlp])
    cv_scores_model[i].extend([cv_score, cv_score_gb, cv_score_dt, cv_score_mlp])

# Create bar graphs for each model
# Create bar graphs for each model
for i in range(3):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # SMAPE Scores
    axes[0].barh(['Random Forest', 'Gradient Boosting', 'Decision Tree', 'MLP'], smape_scores_model[i], color='blue')
    axes[0].set_title(f'Model {i + 1} - SMAPE Scores')
    axes[0].set_xlabel('SMAPE')

    # Accuracy Scores
    axes[1].barh(['Random Forest', 'Gradient Boosting', 'Decision Tree', 'MLP'], accuracy_scores_model[i], color='green')
    axes[1].set_title(f'Model {i + 1} - Accuracy Scores')
    axes[1].set_xlabel('Accuracy')

    # Cross-Validation Scores
    axes[2].barh(['Random Forest', 'Gradient Boosting', 'Decision Tree', 'MLP'], cv_scores_model[i], color='orange')
    axes[2].set_title(f'Model {i + 1} - Cross-Validation Scores')
    axes[2].set_xlabel('Cross-Validation Score')

    plt.tight_layout()
    plt.show()

