import tensorflow as tf
from utils import *

class BiGAN():

	def __init__(input_h, input_w, output_h=64, output_w=64, 
		z_dim = 100, df_dim = 64, gf_dim = 64, c_dim = 3, batch_size = 100):

		self.input_h = input_h
		self.input_w = input_w
		self.output_h = output_h
		self.output_w = output_w
		self.z_dim = z_dim
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.c_dim = c_dim
		self.batch_size = batch_size

	def build():
		self.x_placeholder = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.c_dim])
		self.z_placeholder = tf.placeholder(tf.float32, [self.z_dim])
		self.ex = self.encoder(x_placeholder)
		self.gz = self.decoder() 

	def encoder(self, image, scope_str = "Encoder"):
		with tf.variable_scope(scope_str) as scope:
			conv1 = tf.layers.conv2d(
				inputs = image, 
				filters = self.df_dim, 
				kernel_size = 5, 
				strides = 2,
				name = "conv1")
			lrelu1 = lrelu(conv1, 0.2)
			bn1 = tf.layers.batch_normalization(
				inputs = lrelu1,
				axis = -1,
				name = "bn1")

			conv2 = tf.layers.conv2d(
				inputs = bn1, 
				filters = 2*self.df_dim, 
				kernel_size = 5, 
				strides = 2,
				name = "conv2")
			lrelu2 = lrelu(conv2, 0.2)
			bn2 = tf.layers.batch_normalization(
				inputs = lrelu2,
				axis = -1,
				name = "bn2")


			conv3 = tf.layers.conv2d(
				inputs = bn2, 
				filters = 4*self.df_dim, 
				kernel_size = 5, 
				strides = 2,
				name = "conv3")
			lrelu3 = lrelu(conv3, 0.2)
			bn3 = tf.layers.batch_normalization(
				inputs = lrelu3,
				axis = -1,
				name = "bn3")

			conv4 = tf.layers.conv2d(
				inputs = bn3, 
				filters = 8*self.df_dim, 
				kernel_size = 5, 
				strides = 2,
				name = "conv4")
			lrelu4 = lrelu(conv4, 0.2)
			bn4 = tf.layers.batch_normalization(
				inputs = lrelu4,
				axis = -1,
				name = "bn4")

			bn4_flatten = tf.reshape(bn4, [self.batch_size, -1], name="bn4_flatten")
			out = tf.layers.dense(
				inputs = bn4_flatten,
				units = 1,
				actication = tf.sigmoid(),
				name = "out")
		return out


	def generator(self, z, scope_str = "Generator"):
		first_w = int(self.output_w / 16)
		first_h = int(self.output_h / 16)
		with tf.variable_scope(scope_str) as scope:
			z0 = tf.layers.dense(				
				inputs = z,
				units = first_h*first_w*self.gf_dim*8,
				actication = tf.sigmoid,
				name = "z0")
			z0_reshape = tf.reshape(z0, [-1, first_h, first_w, self.gf_dim*8], name="z0_reshape")
			h = tf.nn.relu(z0_reshape, name="h")

			deconv1 = tf.layers.conv2d_transpose(
				inputs = h,
				filters = self.gf_dim*4,
				kernel_size=5,
				strides=2,
				padding = "same",
				name="deconv1")
			bn1 = tf.layers.batch_normalization(
				inputs = deconv1,
				axis = -1,
				name="bn1")
			relu1 = tf.nn.relu(bn1, name="relu1")

			deconv2 = tf.layers.conv2d_transpose(
				inputs = relu1,
				filters = self.gf_dim*2,
				kernel_size=5,
				strides=2,
				padding = "same",
				name="deconv2")
			bn2 = tf.layers.batch_normalization(
				inputs = deconv2,
				axis = -1,
				name="bn2")	
			relu2 = tf.nn.relu(bn2, name="relu2")	

			deconv3 = tf.layers.conv2d_transpose(
				inputs = relu2,
				filters = self.gf_dim*1,
				kernel_size=5,
				strides=2,
				padding = "same",
				name="deconv3")
			bn3 = tf.layers.batch_normalization(
				inputs = deconv3,
				axis = -1,
				name="bn3")				
			relu3 = tf.nn.relu(bn3, name="relu3")

			deconv4 = tf.layers.conv2d_transpose(
				inputs = relu3,
				filters = self.c_dim,
				kernel_size=5,
				strides=2,
				padding = "same",
				name="deconv4")

			out = tf.tanh(deconv4, name="out")

		return out	




