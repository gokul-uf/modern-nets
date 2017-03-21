'''
Keras implementation of the Original GAN Network based off https://github.com/goodfeli/adversarial/blob/master/mnist.yaml

Generator:
Input dim = 1x100
then followed by a ReLU of output size 1200x1
then ReLU of 1200 size
then sigmoid of output size 784, all layers initialized with uniform noise between [-0.05, +0.05]

Discriminator:
	Input dim: 784
	then Maxout of 240 Linear units with dim 5 each
	then maxout of 240 Linear units with dim 5 each
	then a sigmoid of output 1

Training:
	Number of Iterations: 50000
	Method: Minibatch SGD, Momentum
	initial Momentum: 0.5
	initial LR: 0.1
	batch size: 100
'''
import keras 
import numpy as np
from keras.layers import Input, Dense, MaxoutDense, Dropout, Flatten
from keras.models import Model
from keras.initializations import uniform
from keras.optimizers import SGD
from keras.datasets import mnist
import cPickle as pkl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 100
num_epoch = 1000

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5)/127.5

real_data = np.vstack((X_train, X_test))
noise_input = np.random.randn(real_data.shape[0], 100)

numBatches = real_data.shape[0]/batch_size

print "Shape of real data: {}".format(real_data.shape)
print "Shape of noise input: {}".format(noise_input.shape)
print "Total Epochs: {}".format(num_epoch)
print "Batch Size: {}".format(real_data.shape[0] / numBatches)
print "Number of Batches: {}".format(numBatches)

print "Generator"
input = Input(shape = (100,))
generator = Dense(1200, activation='relu', init = 'uniform')(input)
generator = Dense(1200, activation='relu', init = 'uniform')(generator)
generator = Dense(784, activation='sigmoid', init = 'uniform')(generator)
sgd = SGD(lr=0.1, momentum=0.5, decay=1.000004)
generator = Model(input=input, output=generator)
generator.summary()

print "Discriminator"
d_input = Input(shape = (28, 28))
discriminator = Flatten()(d_input)
discriminator = MaxoutDense(240, nb_feature=5, init='uniform')(discriminator)
discriminator = MaxoutDense(240, nb_feature=5, init='uniform')(discriminator)
discriminator = Dense(1, activation='sigmoid', init='uniform')(discriminator)
discriminator = Model(input=d_input, output=discriminator)
discriminator.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
discriminator.summary()

discriminator.trainable = False

print "GAN"
gan_input = Input(shape = (100,))
H = generator(gan_input)
GAN = discriminator(H)
GAN = Model(input = gan_input, output = GAN)
GAN.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
GAN.summary()

discriminator.trainable = True

print "Training the GAN"
noise_labels = np.zeros(batch_size)
real_labels = np.ones(batch_size)
all_labels = np.concatenate((real_labels, noise_labels))

print "noise_labels: {}".format(noise_labels.shape)
print "real_labels: {}".format(real_labels.shape)
print "all_labels: {}".format(all_labels.shape)

gan_train_loss = []
gan_train_acc = []

disc_train_loss = []
disc_train_acc = []

for i in range(num_epoch):
	print "Epoch: {}".format(i+1)
	np.random.shuffle(real_data)
	np.random.shuffle(noise_input)
	disc_loss = []
	disc_acc = []
	gan_loss = []
	gan_acc = []

	for j in range(numBatches):
#		print "Has to be all: {} or twice".format(batch_size)

		noise_batch = noise_input[i*batch_size: (i+1)*batch_size]
#		print "noise_batch: {}".format(noise_batch.shape)

		noise_data = generator.predict(noise_batch).reshape((batch_size, 28, 28))
#		print "noise_data: {}".format(noise_data.shape)

		train_data = real_data[i*batch_size: (i+1)*batch_size]
#		print "train_data: {}".format(train_data.shape)

		all_data = np.concatenate((train_data, noise_data))
#		print "all_data: {}".format(all_data.shape)

		discLoss, discAcc = discriminator.train_on_batch(all_data, all_labels)
		discriminator.trainable = False
		
		ganLoss, ganAcc = GAN.train_on_batch(noise_batch, np.ones(noise_batch.shape[0]))
		discriminator.trainable = True

		disc_loss.append(discLoss)
		disc_acc.append(discAcc)

		gan_loss.append(ganLoss)
		gan_acc.append(ganAcc)

	print "Average Discriminator Loss: {}, Acc: {}".format(np.average(disc_loss), np.average(disc_acc))
	print "Average GAN Loss: {}, Acc: {}".format(np.average(gan_loss), np.average(gan_acc))

	gan_train_loss.append(np.average(gan_loss))
	disc_train_loss.append(np.average(disc_loss))

	gan_train_acc.append(np.average(gan_acc))
	disc_train_acc.append(np.average(disc_acc))

	if i % 5 == 0:
		test_noise = np.random.randn(1, 100)
		generated_image = generator.predict(test_noise).reshape((28,28))
		plt.figure()
		plt.imshow(generated_image, interpolation='nearest', cmap = "gray_r")
		plt.savefig("epoch_{}.png".format(i))
		plt.close()
	
GAN.save("gan.h5")
discriminator.save("discriminator.h5")
generator.save("generator.h5")

fig = plt.figure()
d, = plt.plot(range(1, len(disc_train_loss)+1), disc_train_loss, label = "Discriminator Loss")
g, = plt.plot(range(1, len(gan_train_loss)+1), gan_train_loss, label = "GAN loss")
fig.legend((d,g), ("D", "G"), "upper right")
plt.savefig("loss.png")

pkl.dump(gan_train_loss, open("gan_train_loss.pkl", "wb"))
pkl.dump(disc_train_loss, open("disc_train_loss.pkl", "wb"))
pkl.dump(gan_train_acc, open("gan_train_acc.pkl", "wb"))
pkl.dump(disc_train_acc, open("disc_train_acc.pkl", "wb"))
