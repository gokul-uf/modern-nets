import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense, Dropout, MaxoutDense, Input, Merge
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.activations import relu
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras.optimizers import adam
try:
	import tqdm
except:
	print "TQDM not found, per epoch progress will not be shown"

NUM_EPOCH = 20000
BATCH_SIZE = 32
LR = 0.1
M = 0.5

def generator():
	noise_input = Input((100,))
	label_input = Input((10,))
	noise = Dropout(0.5)(noise_input)
	noise = Dense(200, activation = 'relu')(noise)
	label = Dropout(0.5)(label_input)
	label = Dense(1000, activation = 'relu')(label)
	G = Concatenate()([noise, label])
	G = Dropout(0.5)(G)
	G = Dense(1200, activation = 'relu')(G)
	G = Dropout(0.5)(G)
	G = Dense(784, activation = 'sigmoid')(G)
	G = Model([noise_input, label_input], G)
	G.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])	
	return G

def discriminator():
	image_input = Input((784,))
	label_input = Input((10,))
	input = Dropout(.5)(image_input)
	input = MaxoutDense(240, 5)(input)
	label = Dropout(.5)(label_input)
	label = MaxoutDense(50, 5)(label)	
	D = Concatenate()([input, label])
	D = Dropout(0.5)(D)
	D = MaxoutDense(240, 4)(D)
	D = Dropout(0.5)(D)
	D = Dense(1, activation = 'sigmoid')(D)
	D = Model([image_input, label_input], D)
	D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return D

def freeze_layers(model, val):
	for layer in model.layers:
		layer.trainable = False

def GAN(discriminator, generator):
	discriminator.trainable = False
	discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print "in GAN"
	discriminator.summary()
	gan_noise_input = Input((100,))
	gan_label_input = Input((10,))
	temp = generator([gan_noise_input, gan_label_input])
	gan = discriminator([temp, gan_label_input])
	gan = Model([gan_noise_input, gan_label_input], gan)
	gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return gan

def get_one_hot(num):
	one_hot = np.zeros(10,)
	one_hot[num] = 1
	return one_hot

def init_data():
	(train_data, train_label), (test_data, test_label) = mnist.load_data()
	data = np.vstack((train_data, test_data))
	data = np.reshape(data, (70000, 784))
	data = (data - 127.5) / 127.5
	temp_labels = np.concatenate((train_label, test_label))
	labels = np.zeros((70000, 10))
	for i in range(len(temp_labels)):
		labels[i] = get_one_hot(temp_labels[i])
	return data, labels


if __name__ == '__main__':
	real_data, real_data_class = init_data()
	print real_data.shape
	print real_data_class.shape
	print "stacking"
	real_data_n_class = np.hstack((real_data, real_data_class))
	print real_data_n_class.shape
	gen = generator()
	disc = discriminator()
	print "Discriminator"
	disc.summary()
	print "generator"
	gen.summary()
	
	print "Freezing Disc"
	freeze_layers(disc, False)
	disc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	gan = GAN(disc, gen)
	gan.summary()

	discriminator_labels = np.asarray([1]*32 + [0]*32)
	gan_labels = np.asarray([1]*32)
	for i in range(NUM_EPOCH):
		disc_epoch_loss = []
		disc_epoch_acc = []
		gan_epoch_loss = []
		gan_epoch_acc = []

		np.random.shuffle(real_data_n_class)
		for j in range(len(real_data) / BATCH_SIZE): 
			start_index = j*BATCH_SIZE
			batch_data_n_class = real_data_n_class[start_index: start_index + BATCH_SIZE]
			batch_data = batch_data_n_class[:, :784]
			batch_class = batch_data_n_class[:, 784:]
			
			batch_noise = np.random.uniform(low = 0.5, high = 0.5, size = (BATCH_SIZE, 100))
			generator_data = gen.predict([batch_noise, batch_class])
			
			discriminator_data = np.vstack((batch_data, generator_data))
			discriminator_class = np.concatenate((batch_class, batch_class), axis = 0)
			d_loss, d_acc = disc.train_on_batch([discriminator_data, discriminator_class], discriminator_labels)
			disc_epoch_loss.append(d_loss)
			disc_epoch_acc.append(d_acc)
			
			# gan_data = np.vstack((batch_data, generator_data))
			# gan_class = np.concatenate((batch_class, batch_class), axis = 0)
			gan_loss, gan_acc = gan.train_on_batch([batch_noise, batch_class], gan_labels) #CHECK
			gan_epoch_loss.append(gan_loss)
			gan_epoch_acc.append(gan_acc)
		
		print "Epoch: {}".format(i)
		print "G Loss: {}, G Acc: {}".format(np.mean(gan_epoch_loss), np.mean(gan_epoch_acc))
		print "D Loss: {}, D Acc: {}".format(np.mean(disc_epoch_loss), np.mean(disc_epoch_acc))		


	# a noise prior z with dimensionality 100 was drawn from a uniform distribution
	# within the unit hypercube
