# All the util functions should be here
import tensorflow as tf

def lrelu(x, alpha):
	return tf.maximum(alpha * x, x)

def residuleBlock(inputs, filters, \
	kernel_size, stride, idx, scope_str = "rB"):
	with tf.variable_scope(scopr_str + idx) as scope:
		