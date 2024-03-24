# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

# new utils
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


# make sure to get the Example Dataset with pip install palmerpenguins
from palmerpenguins import load_penguins

# to visualize the column transformer and pipeline
set_config(display='diagram')

# Pickle Module for Showcase
import pickle

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')


# Grams to Kilograms function
def g_to_kg(g):
    return g/ 1000


# Read in the Data and look at it
penguins = load_penguins()


# 2 rows missing Data nearly completely - we just drop them, when bill_lenght_mm is empty
penguins.dropna(subset = "bill_length_mm", inplace=True)


# Extract Target and Features
y = penguins["species"]

X = penguins[['island', 'bill_length_mm', 'bill_depth_mm',
       'flipper_length_mm', 'body_mass_g', 'sex', 'year']]


# Train Test Split before our Transformations to Avoid Data Leakage

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= .2, random_state=88)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Group Features to Numerical and Categorical
cat_cols = X_train.select_dtypes(include="object").columns
num_cols = X_train.select_dtypes(include="number").columns


# Lets define a pipeline to impute and Encode Sex
# Fill missing values in Sex with the most frequent and Encode the column

sex_pipeline = Pipeline(steps =
                        [("imputer", SimpleImputer(strategy= "most_frequent")),
                         ("sex_ohe", OneHotEncoder(drop="first"))
                         ])

# define our pipeline for bodymass transformations
mass_pipeline = Pipeline(steps = 
                         [("converter", FunctionTransformer(g_to_kg)),
                          ("binning", KBinsDiscretizer())
                          ])

# include the mass pipeline in our transformer - we name it transformer_3 here

transformer_3 = [("sex_pipeline", sex_pipeline, ["sex"]),
                 ("mass_pipeline", mass_pipeline, ["body_mass_g"]),
                 ("ohe", OneHotEncoder(drop="first"), ["island"]),
                 ("scaler", RobustScaler(), ["bill_length_mm", "flipper_length_mm"])
                 ]

# make our new Column Transformer instance

column_transformer_3 = ColumnTransformer(transformer_3,
                                         remainder="drop")

# Fit and Transform the Column(Transformer) ONLY on Training Data
X_train_fe = column_transformer_3.fit_transform(X_train, y_train)

# Use it to ONLY Transform the test Data
X_test_fe = column_transformer_3.transform(X_test)


# nest the Model into Pipeline for quick Processing and Modelling

log_reg_pipeline = Pipeline(steps = 
                            [("column_transformer", column_transformer_3),
                             ("log_reg", LogisticRegression(class_weight="balanced"))
                             ])

# Fit The Model with our Pipeline
log_reg_pipeline.fit(X_train, y_train)


# do the same with random Forest Classifier in the end
forest_pipeline = Pipeline(steps = 
                           [("column_transformer", column_transformer_3),
                            ("forest", RandomForestClassifier(n_estimators=15, max_depth=5))
                           ])


# Fit The Model with our Pipeline
forest_pipeline.fit(X_train, y_train)

# do the same again for a SVC Model
svc_pipeline = Pipeline(steps=
                        [("column_transformer", column_transformer_3),
                         ("svc", SVC(probability=True))
                         ])

# Fit the SVC Model
svc_pipeline.fit(X_train,y_train)

# Dictionary of my Model Proba Predictions
models_proba = {
    "LogisticRegression": log_reg_pipeline.predict_proba(X_test),
    "Random Forrest": forest_pipeline.predict_proba(X_test),
    "SVC": svc_pipeline.predict_proba(X_test)
}

models_pred = {
    "LogisticRegression": log_reg_pipeline.predict(X_test),
    "Random Forrest": forest_pipeline.predict(X_test),
    "SVC": svc_pipeline.predict(X_test)
}


# Compute and print ROC AUC scores
for model_name, y_pred_proba in models_proba.items():
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f'ROC AUC score for {model_name}: {roc_auc:.2f}')

# Compute Accuracy Score for each Model and Store it in Dict for Plot
accuracies = {}
for model_name, y_pred in models_pred.items():
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    print(f"Acurracy Score for {model_name}:", accuracy)

# Plot the Accuracies of my Different Models
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Score of Different Models')
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
plt.show()



