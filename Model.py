import tensorflow as tf
from utils import *
import time
import sys

class BiGAN(object):

	def __init__(self, input_h=64, input_w=64, output_h=64, output_w=64,\
		z_dim = 100, df_dim = 64, gf_dim = 64, c_dim = 3, batch_size = 128,\
		miu = 10, lamb = 500, lr = 2e-4,\
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

		self.miu = miu
		self.lamb = lamb
		self.lr = lr

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
		self.x_reconstruction_loss = tf.reduce_mean((self.x_placeholder-self.gex)**2)
		self.z_reconstruction_loss = tf.reduce_mean((self.z_placeholder-self.egz)**2)

		self.e_loss = -self.divergence_loss + self.miu * self.x_reconstruction_loss
		self.g_loss = self.divergence_loss + self.lamb * self.z_reconstruction_loss

		# add optimizer
		all_trainables = tf.trainable_variables()
		self.e_vars = [var for var in all_trainables if "Encoder" in var.name]
		self.g_vars = [var for var in all_trainables if "Generator" in var.name]
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.e_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)\
			.minimize(self.e_loss, var_list=self.e_vars, global_step=self.global_step)
		self.g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)\
			.minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)

		# add summary operation
		self.gz_summary = tf.summary.image("generated image", self.gz[:2])
		self.x_summary = tf.summary.image("real image", self.x_placeholder[:2])
		self.gex_summary = tf.summary.image("reconstructed image", self.gex[:2])
		self.e_loss_summary = tf.summary.scalar("encoder loss", self.e_loss)
		self.g_loss_summary = tf.summary.scalar("generator loss", self.g_loss)
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

	def run_epoch(self, X_train, X_val):
		N = X_train.shape[0]
		data_batches = getMiniBatch(X_train, batch_size = self.batch_size)
		counter = 0
		for data_batch in data_batches:
			latent_batch = self.latent(self.batch_size)
			d = {self.x_placeholder:data_batch, self.z_placeholder:latent_batch}
			self.sess.run([self.g_optimizer], feed_dict=d)

			self.sess.run([self.g_optimizer], feed_dict=d)

			self.sess.run([self.e_optimizer], feed_dict=d)

			if(counter % 50 == 0):
				print("processing: ", (100.0 * counter * self.batch_size / N, "%"))
				sys.stdout.flush()
				summary, global_step = self.sess.run([self.merged_summary, self.global_step], feed_dict=d)
				self.summary_writer.add_summary(summary, global_step)

			counter += 1


	def divergence(self, e):
		mean, s = tf.nn.moments(e, axes=[0])
		M = e.shape.as_list()[0]
		re = tf.reduce_sum(-0.5+(mean**2+s**2)/2.0-tf.log(s))
		return re

	def latent(self, batch_size):
		z = np.random.randn(batch_size, self.z_dim)
		z /= np.linalg.norm(z, axis=1, keepdims=True)
		return z

	def encoder(self, image, reuse = False, scope_str = "Encoder"):
		with tf.variable_scope(scope_str) as scope:
			if(reuse):
				scope.reuse_variables()
			conv1 = tf.layers.conv2d(
				inputs = image, 
				filters = self.df_dim, 
				kernel_size = 5, 
				strides = 2,
				padding = "same",
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
				padding = "same",
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
				padding = "same",
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
				padding = "same",
				name = "conv4")
			lrelu4 = lrelu(conv4, 0.2)
			bn4 = tf.layers.batch_normalization(
				inputs = lrelu4,
				axis = -1,
				name = "bn4")

			s = bn4.shape.as_list()
			bn4_flatten = tf.reshape(bn4, [-1, s[1]*s[2]*s[3]], name="bn4_flatten")
			dense = tf.layers.dense(
				inputs = bn4_flatten,
				units = self.z_dim,
				name = "dense")
			out = tf.nn.l2_normalize(dense, 1, name="out")
		return out


	def generator(self, z, reuse = False, scope_str = "Generator"):
		first_w = int(self.output_w / 16)
		first_h = int(self.output_h / 16)
		with tf.variable_scope(scope_str) as scope:
			if(reuse):
				scope.reuse_variables()
			z0 = tf.layers.dense(				
				inputs = z,
				units = first_h*first_w*self.gf_dim*8,
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

def main():
	model = BiGAN()
	data_dir = "../data"
	n_epochs = 5
	print("load data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir, prefix="")
	X_train = scale(X_train)
	print("finish loading")
	model.train(X_train, y_train, n_epochs)

if __name__ == "__main__":
	main()

