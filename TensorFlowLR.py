import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd #Giver mulighed for let at manipulere data.
import numpy as np
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers

np.set_printoptions(precision=3, suppress=True)

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

#Turning Data into Feature Columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # Den får altså listen af alle unikke værdier for et given feature column. 
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


#The Input Function:
def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32):
    def input_function(): #Det her er funktionen den returnerer. 
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))# Laver tf.data.Dataset objektet med data og labels. 
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain,y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#Creating The Model:
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) 

#Training The Model:
linear_est.train(train_input_fn) #Træner på vores Træningsdata
result = linear_est.evaluate(eval_input_fn)

os.system('cls')
print(result['accuracy'])





