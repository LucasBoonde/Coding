from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SpealLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#Load Dataset:
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0) #header=0 betyder bare at row 0 er header
test = pd.read_csv(test_path,names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')
print(train.head())

#Input Funciton:
def input_fn(features, labels, training=True, batch_size=256):
    #Konveterer data til dataset:
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #Shuffle hvis det er træningsdata:
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

#Feature Columns:
my_feature_columns =[]
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

#Building The Model:
#tf.get_logger().setLevel('INFO') #Giver output mens den træner DNN.

classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=[30,10], n_classes=3)

classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),steps=5000)

#Evaluate Estimator:
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set Accuracy: {accuracu:0.3f}\n'.format(**eval_result))