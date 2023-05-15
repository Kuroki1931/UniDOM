import sys
sys.path.insert(0, './')

import tensorflow as tf


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))