import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG


class CLS_SSG_Model_PARA(Model):

	def __init__(self, batch_size, action_size, bn=False, activation=None):
		super(CLS_SSG_Model_PARA, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.action_size = action_size
		self.bn = bn
		self.keep_prob = 0.5

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.layer1 = Pointnet_SA(
			npoint=512, radius=0.02,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.layer2 = Pointnet_SA(
			npoint=128,
			radius=0.02,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.layer3 = Pointnet_SA(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, 1024],
			group_all=True,
			activation=self.activation,
			bn = self.bn
		)

		self.dense1 = Dense(256, activation=self.activation)
		self.batchnorm1 = BatchNormalization()
		self.dropout1 = Dropout(self.keep_prob)

		self.parameters_process1 = Dense(256, activation=self.activation)
		self.parameters_process1 = BatchNormalization()

		self.last_dense1 = Dense(128, activation=self.activation)
		self.last_dense2 = Dense(self.action_size, activation=self.activation)


	def forward_pass(self, inputs, training, batch_size=None):
		point, vector_encode, parameters = inputs
		xyz, points = self.layer1(point, vector_encode, training=training)
		xyz, points = self.layer2(xyz, points, training=training)
		xyz, points = self.layer3(xyz, points, training=training)

		if batch_size:
			net = tf.reshape(points, (batch_size, -1))
		else:
			net = tf.reshape(points, (self.batch_size, -1))

		net = self.dense1(net)
		net = self.batchnorm1(net)
		net = self.dropout1(net)

		l = self.parameters_process1(parameters)
		l = self.parameters_process1(l)

		net = tf.concat([net, l], axis=1)

		net = self.last_dense1(net)
		pred = self.last_dense2(net)

		return pred

	def train_step(self, input):

		with tf.GradientTape() as tape:

			pred = self.forward_pass([input[0], input[1], input[2]], True)
			loss = self.compiled_loss(input[3], pred)
		
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(
			zip(gradients, self.trainable_variables))

		self.compiled_metrics.update_state(input[3], pred)

		return {m.name: m.result() for m in self.metrics}


	def test_step(self, input):

		pred = self.forward_pass([input[0], input[1], input[2]], False)
		loss = self.compiled_loss(input[3], pred)

		self.compiled_metrics.update_state(input[3], pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, inputs, training=False):

		return self.forward_pass(inputs, training)


class CLS_SSG_Model(Model):

	def __init__(self, batch_size, action_size, bn=False, activation=None):
		super(CLS_SSG_Model, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.action_size = action_size
		self.bn = bn
		self.keep_prob = 0.5

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.layer1 = Pointnet_SA(
			npoint=512, radius=0.02,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.layer2 = Pointnet_SA(
			npoint=128,
			radius=0.02,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.layer3 = Pointnet_SA(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, 1024],
			group_all=True,
			activation=self.activation,
			bn = self.bn
		)

		self.dense1 = Dense(256, activation=self.activation)
		self.batchnorm1 = BatchNormalization()
		self.dropout1 = Dropout(self.keep_prob)

		self.last_dense1 = Dense(128, activation=self.activation)
		self.last_dense2 = Dense(self.action_size, activation=self.activation)


	def forward_pass(self, inputs, training, batch_size=None):
		point, vector_encode = inputs
		xyz, points = self.layer1(point, vector_encode, training=training)
		xyz, points = self.layer2(xyz, points, training=training)
		xyz, points = self.layer3(xyz, points, training=training)

		if batch_size:
			net = tf.reshape(points, (batch_size, -1))
		else:
			net = tf.reshape(points, (self.batch_size, -1))

		net = self.dense1(net)
		net = self.batchnorm1(net)
		net = self.dropout1(net)

		net = self.last_dense1(net)
		pred = self.last_dense2(net)

		return pred

	def train_step(self, input):

		with tf.GradientTape() as tape:

			pred = self.forward_pass([input[0], input[1]], True)
			loss = self.compiled_loss(input[2], pred)
		
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(
			zip(gradients, self.trainable_variables))

		self.compiled_metrics.update_state(input[2], pred)

		return {m.name: m.result() for m in self.metrics}


	def test_step(self, input):

		pred = self.forward_pass([input[0], input[1]], False)
		loss = self.compiled_loss(input[2], pred)

		self.compiled_metrics.update_state(input[2], pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, inputs, training=False):

		return self.forward_pass(inputs, training)