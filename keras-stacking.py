import tensorflow as tf
from sklearn.datasets import make_blobs
import numpy as np
from numpy import dstack

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# generate 2d classification dataset
import os
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
# one hot encode output variable
y = tf.keras.utils.to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)

#模型构建：参照keras官方文档https://keras.io/zh/getting-started/sequential-model-guide/
# define model
model = tf.keras.models.Sequential()#建立时序模型
model.add(tf.keras.layers.Dense(25, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model，训练模型
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)

# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# fit model on dataset

def fit_model(trainX, trainy):
	# define model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(25, input_dim=2, activation='relu'))
	model.add(tf.keras.layers.Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=0)
	return model


# create directory for models
os.makedirs('models')

# fit and save models
n_members = 5
for i in range(n_members):
	# fit model
	model = fit_model(trainX, trainy)
	# save model
	filename = 'models/model_' + str(i + 1) + '.h5'
	model.save(filename)
	print('>Saved %s' % filename)


# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
		model = tf.keras.models.load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
# load all models
n_members = 5
members = load_all_models(n_members)#所有模型
print('Loaded %d models' % len(members))
#base-line
# evaluate standalone models on test dataset
for model in members:
	# testy_enc = tf.keras.utils.to_categorical(testy)
	_, acc = model.evaluate(testX, testy, verbose=0)
	print('Model Accuracy: %.3f' % acc)


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
	return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	return model

# fit stacked model using the ensemble
model = fit_stacked_model(members, testX, np.argmax(testy, axis=1))
print(testy)
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# evaluate model on test set
yhat = stacked_prediction(members, model, testX)
yhat = tf.keras.utils.to_categorical(yhat)
print(yhat)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
