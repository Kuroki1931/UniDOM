import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG


class Attention(Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


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

		self.plasticine_layer1 = Pointnet_SA(
			npoint=512, radius=0.01,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.plasticine_layer2 = Pointnet_SA(
			npoint=128,
			radius=0.01,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.plasticine_layer3 = Pointnet_SA(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, 1024],
			group_all=True,
			activation=self.activation,
			bn = self.bn
		)

		self.primitive_layer1 = Pointnet_SA(
			npoint=512, radius=0.01,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.primitive_layer2 = Pointnet_SA(
			npoint=128,
			radius=0.01,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn = self.bn
		)

		self.primitive_layer3 = Pointnet_SA(
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

		self.dense2 = Dense(128, activation=self.activation)
		self.batchnorm2 = BatchNormalization()
		self.dropout2 = Dropout(self.keep_prob)

		self.dense3 = Dense(64, activation=self.activation)
		self.batchnorm3 = BatchNormalization()
		self.dropout3 = Dropout(self.keep_prob)

		self.dense4 = Dense(32, activation=self.activation)
		self.batchnorm4 = BatchNormalization()
		self.dropout4 = Dropout(self.keep_prob)

		self.lang_process1 = Dense(512, activation=self.activation)
		self.lang_batchnorm1 = BatchNormalization()

		self.lang_process2 = Dense(256, activation=self.activation)
		self.lang_batchnorm2 = BatchNormalization()

		self.lang_process3 = Dense(128, activation=self.activation)
		self.lang_batchnorm3 = BatchNormalization()

		self.lang_process4 = Dense(64, activation=self.activation)
		self.lang_batchnorm4 = BatchNormalization()

		self.dense5 = Dense(self.action_size, activation=self.activation)


	def forward_pass(self, inputs, training, batch_size=None):
		plasticine_pc, primitive_pc, plasticine_ve, primitive_ve, lang = inputs
		plasticine_xyz, plasticine_points = self.plasticine_layer1(plasticine_pc, plasticine_ve, training=training)
		plasticine_xyz, plasticine_points = self.plasticine_layer2(plasticine_xyz, plasticine_points, training=training)
		plasticine_xyz, plasticine_points = self.plasticine_layer3(plasticine_xyz, plasticine_points, training=training)

		primitive_xyz, primitive_points = self.primitive_layer1(primitive_pc, primitive_ve, training=training)
		primitive_xyz, primitive_points = self.primitive_layer2(primitive_xyz, primitive_points, training=training)
		primitive_xyz, primitive_points = self.primitive_layer3(primitive_xyz, primitive_points, training=training)

		if batch_size:
			plasticine_points = tf.reshape(plasticine_points, (batch_size, -1))
			primitive_points = tf.reshape(primitive_points, (batch_size, -1))
			net = tf.concat([plasticine_points, primitive_points], axis=1)
		else:
			plasticine_points = tf.reshape(plasticine_points, (self.batch_size, -1))
			primitive_points = tf.reshape(primitive_points, (self.batch_size, -1))
			net = tf.concat([plasticine_points, primitive_points], axis=1)

		net = self.dense1(net)
		net = self.batchnorm1(net)
		net = self.dropout1(net)

		net = self.dense2(net)
		net = self.batchnorm2(net)
		net = self.dropout2(net)

		net = self.dense3(net)
		net = self.batchnorm3(net)
		net = self.dropout3(net)

		net = self.dense4(net)
		net = self.batchnorm4(net)
		net = self.dropout4(net)

		l = self.lang_process1(lang)
		# l = self.lang_batchnorm1(l)
	
		l = self.lang_process2(l)
		# l = self.lang_batchnorm2(l)

		l = self.lang_process3(l)
		# l = self.lang_batchnorm1(l)
	
		l = self.lang_process4(l)
		# l = self.lang_batchnorm2(l)

		net = tf.concat([net, l], axis=1)

		pred = self.dense5(net)

		return pred


	def train_step(self, input):

		with tf.GradientTape() as tape:

			pred = self.forward_pass([input[0], input[1], input[2], input[3], input[4]], True)
			loss = self.compiled_loss(input[5], pred)
		
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(
			zip(gradients, self.trainable_variables))

		self.compiled_metrics.update_state(input[5], pred)

		return {m.name: m.result() for m in self.metrics}


	def test_step(self, input):

		pred = self.forward_pass([input[0], input[1], input[2], input[3], input[4]], False)
		loss = self.compiled_loss(input[5], pred)

		self.compiled_metrics.update_state(input[5], pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, inputs, training=False):

		return self.forward_pass(inputs, training)
