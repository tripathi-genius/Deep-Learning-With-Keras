# -*- coding: utf-8 -*-

#Importing modules for data
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras
import matplotlib.pyplot as plt

#Data loading
flower_data = pd.read_csv('iris.csv')
print(flower_data.head())

#Basic Preprocessing of data
encoder = preprocessing.LabelEncoder()
flower_data['Species'] = encoder.fit_transform(flower_data['Species'])
species = flower_data.to_numpy()
YFlowerSpecies = species[:, 4]
XFlowerSpecifications = species[:, 0:4]

#Viewing Y values
print(YFlowerSpecies)

#Viewing X values
print(XFlowerSpecifications)

#Scaling the models
scale_balancer = StandardScaler().fit(XFlowerSpecifications)
SpecificationOfFlowers = scale_balancer.fit_transform(XFlowerSpecifications)
print(SpecificationOfFlowers)

#Categorical transformation for the Y data
catSpecies = tf.keras.utils.to_categorical(YFlowerSpecies, 3)
print(catSpecies)

#Splitting training and testing data
XSpecificationTrain, XSpecificationTest, YSpeciesTrain, YSpeciesTest = train_test_split(SpecificationOfFlowers, catSpecies, test_size=0.10)
#Training data: XSpecificationTrain, YSpecificationTrain
#Testing data: XSpecificationTest, YSpecificationTest

#Prepare a deep learning model
targetClasses = 3
model = tf.keras.Sequential()
model.add(keras.layers.Dense(128, 
                             input_shape=(4, ), 
                             name="my_layer_1",
                             activation='relu'))
model.add(keras.layers.Dense(128,
                             input_shape=(4, ),
                             name="my_layer_2", 
                             activation='relu'))
model.add(keras.layers.Dense(targetClasses,
                             name='OutputClass'))
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

#Training the model for evaluation

verbose_ = 1
batch = 16
epoch = 10
valSplit = 0.2

print("\nModel traing in progress:\n---------------------------")
history = model.fit(XSpecificationTrain,
                    YSpeciesTrain,
                    batch_size=batch, 
                    epochs=epoch,
                    verbose=verbose_,
                    validation_split=valSplit)
print("\nAccuracy during training:\n---------------------------")
pd.DataFrame(history.history)['accuracy'].plot(figsize=(8,5))
plt.title("Accuracy improvement with epoch")
plt.show()
print("\nEvaluation against test dataset\n---------------------------")
model.evaluate(XSpecificationTrain,YSpeciesTrain)


#Save model
model.save("my_iris")

loaded_model = keras.models.load_model("my_iris")
loaded_model.summary()
prediction_input = [[6.6, 3. , 4.4, 1.4]]
scaled_input = scale_balancer.transform(prediction_input)
raw_prediction = model.predict(scaled_input)

print("Raw Prediction Output (Probabilities) :" , raw_prediction)
prediction = np.argmax(raw_prediction)
print("Prediction is ", encoder.inverse_transform([prediction]))

