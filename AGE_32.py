import tensorflow as tf
from utils import *
import time
import sys
from tensorflow.examples.tutorials.mnist import input_data

class AGE_32(object):

	def __init__(self, input_h=32, input_w=32, output_h=32, output_w=32,\
		z_dim = 64, df_dim = 64, gf_dim = 64, c_dim = 3, batch_size = 128, total_N = 55000, \
		miu = 10, lamb = 1000, lr = 2e-4, decay_every = 5, g_iter = 2, \
		log_dir = "../log", save_dir = "../check_points"):

		self.input_h = input_h
		self.input_w = input_w
		self.output_h = output_h
		self.output_w = output_w
		self.z_dim = z_dim
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.c_dim = c_dim
		self.batch_size = batch_size
		self.total_N = 55000

		self.miu = miu
		self.lamb = lamb
		self.lr = lr
		self.decay_every = decay_every
		self.g_iter = g_iter

		self.log_dir = log_dir
		self.save_dir = save_dir

	def build(self):
		# add placeholder for image x and latent variable z
		self.x_placeholder = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.c_dim], name="x_placeholder")
		self.z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name="z_placeholder")

		# add the transformed tensor
		self.ex = self.encoder(self.x_placeholder)
		self.gex = self.generator(self.ex)
		self.gz = self.generator(self.z_placeholder, reuse = True)
		self.egz = self.encoder(self.gz, reuse = True)

		# add losses
		self.divergence_loss = self.divergence(self.egz) - self.divergence(self.ex)
		self.x_reconstruction_loss = tf.reduce_mean(tf.abs(self.x_placeholder-self.gex))
		self.z_reconstruction_loss = 1 - tf.reduce_mean(self.z_placeholder*self.egz)

		self.e_loss = -self.divergence_loss + self.miu * self.x_reconstruction_loss
		self.g_loss = self.divergence_loss + self.lamb * self.z_reconstruction_loss

		# add optimizer
		all_trainables = tf.trainable_variables()
		self.e_vars = [var for var in all_trainables if "Encoder" in var.name]
		self.g_vars = [var for var in all_trainables if "Generator" in var.name]
		print("Encoder variables:")
		for var in self.e_vars:
			print(var.name)
		print("Generator variables:")
		for var in self.g_vars:
			print(var.name)
		self.e_step = tf.Variable(0, name="e_step", trainable=False)
		self.g_step = tf.Variable(0, name='g_step', trainable=False)
		decay_steps = int(self.decay_every) * int(self.total_N / self.batch_size)
		self.decayed_lr = tf.train.exponential_decay(self.lr, 
			self.e_step, decay_steps, 0.5, staircase=True, name="decayed_lr")
		self.e_optimizer = tf.train.AdamOptimizer(self.decayed_lr, beta1=0.5)\
			.minimize(self.e_loss, var_list=self.e_vars, global_step=self.e_step)
		self.g_optimizer = tf.train.AdamOptimizer(self.decayed_lr, beta1=0.5)\
			.minimize(self.g_loss, var_list=self.g_vars, global_step=self.g_step)

		# add summary operation
		self.gz_summary = tf.summary.image("generated image", self.gz)
		self.x_summary = tf.summary.image("real image", self.x_placeholder)
		self.gex_summary = tf.summary.image("reconstructed image", self.gex)
		mean, var = tf.nn.moments(self.egz, axes=[0])
		self.mean_summary = tf.summary.histogram("component-wise mean", mean)
		self.var_summary = tf.summary.histogram("component-wise var", var)
		self.divergence_summary = tf.summary.scalar("divergence loss", self.divergence_loss)
		self.x_reconstruction_loss_summary = tf.summary.scalar("x reconstruction loss", self.miu * self.x_reconstruction_loss)
		self.z_reconstruction_loss_summary = tf.summary.scalar("z reconstruction loss", self.lamb * self.z_reconstruction_loss)
		self.decayed_lr_summary = tf.summary.scalar("decayed learning rate", self.decayed_lr)
		self.merged_summary = tf.summary.merge_all()

		# add session, log writer and saver
		self.sess = tf.Session()
		self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
		self.saver = tf.train.Saver()

	def train(self, X_train, X_val, epochs):
		print("build models...")
		s = time.time()
		self.build()
		print("finish building, using ", time.time()-s, " seconds")
		self.sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			print("training for epoch ", i)
			self.run_epoch(X_train, X_val)
		self.saver.save(self.sess, self.save_dir, global_step=self.e_step)
		print("Model saved at", self.save_dir)

	def run_epoch(self, X_train, X_val):
		N = X_train.shape[0]
		data_batches = getMiniBatch(X_train, batch_size = self.batch_size)
		counter = 0
		for data_batch in data_batches:
			latent_batch = self.latent(self.batch_size)
			d = {self.x_placeholder:data_batch, self.z_placeholder:latent_batch}
			for i in range(self.g_iter - 1):
				self.sess.run([self.g_optimizer], feed_dict=d)

			zr, _, gs= self.sess.run([self.z_reconstruction_loss_summary, self.g_optimizer, self.g_step], feed_dict=d)

			xr, ds, ms, vs, lrs, _, es = self.sess.run([self.x_reconstruction_loss_summary, self.divergence_summary, 
				self.mean_summary, self.var_summary, self.decayed_lr_summary, self.e_optimizer, self.e_step], feed_dict=d)

			if(counter % 10 == 0):
				print("processing: ", (100.0 * counter * self.batch_size / N, "%"))
				sys.stdout.flush()
				a, b, c= self.sess.run([self.x_summary, self.gz_summary, self.gex_summary], feed_dict=d)
				self.summary_writer.add_summary(a, es)
				self.summary_writer.add_summary(b, es)
				self.summary_writer.add_summary(c, es)
			else:
				self.summary_writer.add_summary(zr, gs)
				self.summary_writer.add_summary(xr, es)
				self.summary_writer.add_summary(ds, es)
				self.summary_writer.add_summary(ms, es)
				self.summary_writer.add_summary(vs, es)
				self.summary_writer.add_summary(lrs, es)

			counter += 1


	def divergence(self, e):
		mean, s2 = tf.nn.moments(e, axes=[0])
		re = tf.reduce_mean(-0.5+(mean**2+s2**2)/2.0-tf.log(s2))
		return re

	def latent(self, batch_size):
		z = np.random.randn(batch_size, self.z_dim)
		z /= np.linalg.norm(z, axis=1, keepdims=True)
		return z

	def encoder(self, image, reuse = False, scope_str = "Encoder"):
		with tf.variable_scope(scope_str) as scope:
			if(reuse):
				scope.reuse_variables()
			#input size 32 * 32 * 3
			conv1 = tf.layers.conv2d(
				inputs = image, 
				filters = self.df_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name = "conv1")
			lrelu1 = lrelu(conv1, 0.2)
			bn1 = tf.layers.batch_normalization(
				inputs = lrelu1,
				axis = -1,
				name = "bn1")

			#input size 16 * 16 * df_dim
			conv2 = tf.layers.conv2d(
				inputs = bn1, 
				filters = 2*self.df_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name = "conv2")
			lrelu2 = lrelu(conv2, 0.2)
			bn2 = tf.layers.batch_normalization(
				inputs = lrelu2,
				axis = -1,
				name = "bn2")

			#input size 8 * 8 * 2df_dim
			conv3 = tf.layers.conv2d(
				inputs = bn2, 
				filters = 4*self.df_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name = "conv3")
			lrelu3 = lrelu(conv3, 0.2)
			bn3 = tf.layers.batch_normalization(
				inputs = lrelu3,
				axis = -1,
				name = "bn3")

			#input size 4 * 4 * * 4df_dim
			conv4 = tf.layers.conv2d(
				inputs = bn3, 
				filters = self.z_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = True,
				name = "conv4")

			#input size 2 * 2 * z_dim
			out = tf.layers.average_pooling2d(conv4, 2, 2)

			#input size 1 * 1 * z_dim
			out = tf.reshape(out, [-1, self.z_dim])
			out = tf.nn.l2_normalize(out, 1, name="out")
		return out


	def generator(self, z, reuse = False, scope_str = "Generator"):
		first_w = int(self.output_w / 16)
		first_h = int(self.output_h / 16)
		with tf.variable_scope(scope_str) as scope:
			if(reuse):
				scope.reuse_variables()

			z = tf.reshape(z, [-1, 1, 1, self.z_dim])
			#input size 1 * 1 * z_dim
			deconv1 = tf.layers.conv2d_transpose(
				inputs = z,
				filters = self.gf_dim*8,
				kernel_size=4,
				strides=1,
				padding = "valid",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name="deconv1")
			bn1 = tf.layers.batch_normalization(
				inputs = deconv1,
				axis = -1,
				name="bn1")
			relu1 = tf.nn.relu(bn1, name="relu1")

			#input size 4 * 4 * 8gf_dim
			deconv2 = tf.layers.conv2d_transpose(
				inputs = relu1,
				filters = self.gf_dim*4,
				kernel_size=4,
				strides=2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name="deconv2")
			bn2 = tf.layers.batch_normalization(
				inputs = deconv2,
				axis = -1,
				name="bn2")	
			relu2 = tf.nn.relu(bn2, name="relu2")	

			#input size 8 * 8 * 4gf_dim
			deconv3 = tf.layers.conv2d_transpose(
				inputs = relu2,
				filters = self.gf_dim*2,
				kernel_size=4,
				strides=2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name="deconv3")
			bn3 = tf.layers.batch_normalization(
				inputs = deconv3,
				axis = -1,
				name="bn3")				
			relu3 = tf.nn.relu(bn3, name="relu3")

			#input size 16 * 16 * 2gf_dim
			deconv4 = tf.layers.conv2d_transpose(
				inputs = relu3,
				filters = self.gf_dim*2,
				kernel_size=4,
				strides=2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name="deconv4")
			bn4 = tf.layers.batch_normalization(
				inputs = deconv4,
				axis = -1,
				name="bn4")				
			relu4 = tf.nn.relu(bn4, name="relu4")

			#input size 32 * 32 * 2gf_dim
			conv1 = tf.layers.conv2d(inputs = relu4,
					filters = self.c_dim, 
					kernel_size = 1,
					kernel_initializer = tf.random_normal_initializer(stddev=0.02),
					use_bias = True,
					name = "conv1")

			#input size 32 * 32 * c_dim
			out = tf.tanh(conv1, name="out")

		return out	

def main():
	data = input("what data to use? ")
	n_epochs = 30
	if(data == "svhn"):
		data_dir = "../data_SVHN"
		log_dir = "../SVHN_log"
		save_dir = "../SVHN_check_points"
		print("load data...")
		X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir, prefix="")
		X_train = scale(X_train)
		print("finish loading")
		model = AGE_32(log_dir=log_dir, save_dir=save_dir, g_iter=2, miu=10, lamb=500)
	elif(data == "mnist"):
		log_dir = "../MNIST_log"
		save_dir = "../MNIST_check_points"
		print("load data...")
		mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
		X_train = np.reshape(mnist.train.images, [-1,28,28])
		X_val = np.reshape(mnist.validation.images, [-1,28,28])
		X_train = np.lib.pad(X_train,((0,0),(2,2),(2,2)),'constant',constant_values=((0,0),(0,0),(0,0)))
		X_val = np.lib.pad(X_val,((0,0),(2,2),(2,2)),'constant',constant_values=((0,0),(0,0),(0,0)))
		X_train = np.expand_dims(X_train, 3)
		X_val = np.expand_dims(X_val, 3)
		X_train = scale(X_train)
		print("finish loading")
		model = AGE_32(log_dir=log_dir, save_dir=save_dir, c_dim=1, z_dim=10, lamb=100)

	model.train(X_train, X_val, n_epochs)

if __name__ == "__main__":
	main()

