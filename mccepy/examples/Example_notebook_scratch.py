#!/usr/bin/env python
# coding: utf-8

# ## Use MCCE method without using the Data or RandomForest class

# In[15]:


import warnings
warnings.filterwarnings('ignore')

import sys
import os
# Tricks for loading data and libraries from parent directories. 
grandparent = os.path.abspath('.')
parent = os.path.abspath('./mccepy')
sys.path.insert(1, parent)
sys.path.insert(2, grandparent)
print(os.getcwd())


import re
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mcce.mcce import MCCE
from mcce.metrics import distance, feasibility, constraint_violation, success_rate


# ## Load data

# In[16]:


feature_order = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation', 
                 'relationship', 'race', 'sex', 'hours-per-week',]
                 
dtypes = {"age": "float", 
          "workclass": "category", 
          "fnlwgt": "float", 
          "education-num": "float",
          "marital-status": "category", 
          "occupation": "category", 
          "relationship": "category", 
          "race": "category",
          "sex": "category", 
          "hours-per-week": "float",
          "income": "category"}

categorical = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
continuous = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
immutables = ['age', 'sex']
target = ['income']
features = categorical + continuous

path = 'mccepy/Data/adult_data.csv'

df = pd.read_csv(path)
df = df[features + target]

print(f"The immutable features are {immutables}")


# ## Scale the continuous features between 0 and 1. Encode the categorical features using one-hot encoding

# In[17]:


encoder = preprocessing.OneHotEncoder(drop="first", sparse=False).fit(df[categorical])
df_encoded = encoder.transform(df[categorical])

scaler = preprocessing.MinMaxScaler().fit(df[continuous])
df_scaled = scaler.transform(df[continuous])

categorical_encoded = encoder.get_feature_names_out(categorical).tolist()
df_scaled = pd.DataFrame(df_scaled, columns=continuous)
df_encoded = pd.DataFrame(df_encoded, columns=categorical_encoded)

df = pd.concat([df_scaled, df_encoded, df[target]], axis=1)

print(f"The encoded categorical features are {categorical_encoded}")


# ## Define an inverse_transform function to go easily back to the non-scaled/encoded feature version

# In[18]:


def inverse_transform(df, 
                      scaler, 
                      encoder, 
                      continuous,
                      categorical,
                      categorical_encoded, 
                      ):

    df_categorical = pd.DataFrame(encoder.inverse_transform(df[categorical_encoded]), columns=categorical)
    df_continuous = pd.DataFrame(scaler.inverse_transform(df[continuous]), columns=continuous)

    return pd.concat([df_categorical, df_continuous], axis=1)


# ## Find the immutable features in their encoded form

# In[19]:


immutables_encoded = []
for immutable in immutables:
    if immutable in categorical:
        for new_col in categorical_encoded:
            match = re.search(immutable, new_col)
            if match:
                immutables_encoded.append(new_col)
    else:
        immutables_encoded.append(immutable)

print(f"Encoded immutable features are: {immutables_encoded}")


# ## Create data object to feed into MCCE method

# In[20]:


class Dataset():
    def __init__(self, 
                 immutables, 
                 target,
                 categorical,
                 immutables_encoded,
                 continuous,
                 features,
                 encoder,
                 scaler,
                 inverse_transform,
                 ):
        
        self.immutables = immutables
        self.target = target
        self.feature_order = feature_order
        self.dtypes = dtypes
        self.categorical = categorical
        self.continuous = continuous
        self.features = self.categorical + self.continuous
        self.cols = self.features + [self.target]
        self.immutables_encoded = immutables_encoded
        self.encoder = encoder
        self.scaler = scaler
        self.inverse_transform = inverse_transform
        
dataset = Dataset(immutables, 
                  target,
                  categorical,
                  immutables_encoded,
                  continuous,
                  features,
                  encoder,
                  scaler,
                  inverse_transform)

dtypes = dict([(x, "float") for x in continuous])
for x in categorical_encoded:
    dtypes[x] = "category"
df = (df).astype(dtypes)


# ## Train predictive model

# In[21]:


y = df[target]
X = df.drop(target, axis=1)
test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
clf = RandomForestClassifier(max_depth=None, random_state=0)
ml_model = clf.fit(X_train, y_train)

pred_train = ml_model.predict(X_train)
pred_test = ml_model.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_train, pred_train, pos_label=1)
train_auc = metrics.auc(fpr, tpr)

fpr, tpr, _ = metrics.roc_curve(y_test, pred_test, pos_label=1)
test_auc = metrics.auc(fpr, tpr)

model_prediction = clf.predict(X)

print(f"The out-of-sample AUC is {round(test_auc, 2)}")


# ## Select observations to generate counterfactuals for

# In[22]:


preds = ml_model.predict_proba(df.drop(target, axis=1))[:,1]
factual_id = np.where(preds < 0.5)
factuals = df.loc[factual_id]
test_factual = factuals.iloc[:5]

print(test_factual.head(2))


# ## Fit MCCE method

# In[23]:


mcce = MCCE(dataset=dataset, model=ml_model)

print("Fit trees")
mcce.fit(df.drop(target, axis=1), dtypes)

print("Sample observations for the specific test observations")
cfs = mcce.generate(test_factual.drop(target, axis=1), k=100)

print("Process the sampled observations")
mcce.postprocess(cfs=cfs, test_factual=test_factual, cutoff=0.5, higher_cardinality=False)


# ## Print counterfactuals

# In[24]:


cfs = mcce.results_sparse
cfs['income'] = test_factual['income'] # add back the original response

# invert the features to their original form
print("Original factuals:")
decoded_factuals = dataset.inverse_transform(test_factual,
                                             scaler, 
                                             encoder, 
                                             continuous,
                                             categorical,
                                             categorical_encoded)[feature_order]

decoded_factuals


# In[25]:


print("Generated counterfactuals:")
decoded_cfs = dataset.inverse_transform(cfs,
                                        scaler, 
                                        encoder, 
                                        continuous,
                                        categorical,
                                        categorical_encoded)[feature_order]
decoded_cfs


# In[26]:


print(decoded_cfs)


# ## Calculate some metrics

# In[27]:


# distance_pd = pd.DataFrame(distance(cfs, test_factual, dataset))

# feasibility_pd = pd.DataFrame(feasibility(cfs, df, categorical_encoded + continuous), columns=['feasibility'])

# const_pd = pd.DataFrame(constraint_violation(decoded_cfs, decoded_factuals, dataset), columns=['violation'])

# success_pd = pd.DataFrame(success_rate(cfs[categorical_encoded + continuous], ml_model), columns=['success'])


# # In[28]:


# results = pd.concat([decoded_cfs, distance_pd, feasibility_pd, const_pd, success_pd], axis=1)
# results

