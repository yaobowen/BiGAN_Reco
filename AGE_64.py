import tensorflow as tf
from utils import *
import time
import sys
from tensorflow.examples.tutorials.mnist import input_data
import argparse

class AGE_64(object):

	def __init__(self, input_h=64, input_w=64, output_h=64, output_w=64,\
		z_dim = 64, df_dim = 64, gf_dim = 64, c_dim = 3, batch_size = 64, total_N = 55000, \
		miu = 0, lamb = 0, lr = 2e-4, decay_every = 5, g_iter = 2, \
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
		self.x_placeholder = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name="x_placeholder")
		self.z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name="z_placeholder")
		self.is_training = tf.placeholder(tf.bool)
		#reshape and preprocess the input image
		self.x = tf.image.resize_images(self.x_placeholder, [self.input_h, self.input_w])
		self.x = self.scale(self.x)


		# add the transformed tensor
		self.ex = self.encoder(self.x)
		self.gex = self.generator(self.ex)
		self.gz = self.generator(self.z_placeholder, reuse = True)
		self.egz = self.encoder(self.gz, reuse = True)

		# add losses
		self.fake_divergence = self.divergence(self.egz)
		self.real_divergence = self.divergence(self.ex)
		self.divergence_loss = self.fake_divergence - self.real_divergence
		self.x_reconstruction_loss = tf.reduce_mean(tf.abs(self.x-self.gex))
		self.z_reconstruction_loss = 1 - tf.reduce_mean(self.z_placeholder*self.egz)

		self.e_loss = self.real_divergence - self.fake_divergence + self.miu * self.x_reconstruction_loss
		self.g_loss = self.fake_divergence + self.lamb * self.z_reconstruction_loss

		# add optimizer
		all_trainables = tf.trainable_variables()
		self.e_vars = [var for var in all_trainables if "Encoder" in var.name]
		self.g_vars = [var for var in all_trainables if "Generator" in var.name]
		# print("Encoder variables:")
		# for var in self.e_vars:
		# 	print(var.name)
		# print("Generator variables:")
		# for var in self.g_vars:
		# 	print(var.name)
		self.e_step = tf.Variable(0, name="e_step", trainable=False)
		self.g_step = tf.Variable(0, name='g_step', trainable=False)
		decay_steps = int(self.decay_every) * int(self.total_N / self.batch_size)
		self.decayed_lr = tf.train.exponential_decay(self.lr, 
			self.e_step, decay_steps, 0.5, staircase=True, name="decayed_lr")
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		print(extra_update_ops)
		with tf.control_dependencies(extra_update_ops):
			self.e_optimizer = tf.train.AdamOptimizer(self.decayed_lr, beta1=0.5)\
				.minimize(self.e_loss, var_list=self.e_vars, global_step=self.e_step)
			self.g_optimizer = tf.train.AdamOptimizer(self.decayed_lr, beta1=0.5)\
				.minimize(self.g_loss, var_list=self.g_vars, global_step=self.g_step)

		# add summary operation
		self.gz_summary = tf.summary.image("generated image", self.rescale(self.gz), max_outputs=4)
		self.x_summary = tf.summary.image("real image", self.rescale(self.x), max_outputs=4)
		self.gex_summary = tf.summary.image("reconstructed image", self.rescale(self.gex), max_outputs=4)
		mean, var = tf.nn.moments(self.egz, axes=[0])
		self.mean_summary = tf.summary.histogram("component-wise mean", mean)
		self.var_summary = tf.summary.histogram("component-wise var", var)
		self.real_divergence_summary = tf.summary.scalar("real divergence loss", self.real_divergence)
		self.fake_divergence_summary = tf.summary.scalar("fake divergence loss", self.fake_divergence)
		self.x_reconstruction_loss_summary = tf.summary.scalar("x reconstruction loss", self.miu * self.x_reconstruction_loss)
		self.z_reconstruction_loss_summary = tf.summary.scalar("z reconstruction loss", self.lamb * self.z_reconstruction_loss)
		self.decayed_lr_summary = tf.summary.scalar("decayed learning rate", self.decayed_lr)
		self.merged_summary = tf.summary.merge_all()

		# add session, log writer and saver
		self.sess = tf.Session()
		self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
		self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

	def train(self, X_train, X_val, epochs, restore):
		print("build models...")
		s = time.time()
		self.build()
		print("finish building, using ", time.time()-s, " seconds")
		if(restore):
			self.saver.restore(self.sess, self.save_dir)
		else:
			self.sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			print("training for epoch ", i)
			self.run_epoch(X_train, X_val, i)
		self.saver.save(self.sess, self.save_dir)
		print("Model saved at", self.save_dir)

	def run_epoch(self, X_train, X_val, epoch):
		N = X_train.shape[0]
		data_batches = getMiniBatch(X_train, batch_size = self.batch_size)
		counter = 0
		for data_batch in data_batches:
			latent_batch = self.latent(self.batch_size)
			d = {self.x_placeholder:data_batch, self.z_placeholder:latent_batch, self.is_training:True}
			for i in range(self.g_iter - 1):
				self.sess.run([self.g_optimizer], feed_dict=d)

			zr, _, gs= self.sess.run([self.z_reconstruction_loss_summary, self.g_optimizer, self.g_step], feed_dict=d)

			xr, rds, fds, ms, vs, lrs, _, es = self.sess.run([self.x_reconstruction_loss_summary, self.real_divergence_summary, self.fake_divergence_summary,
				self.mean_summary, self.var_summary, self.decayed_lr_summary, self.e_optimizer, self.e_step], feed_dict=d)

			if(counter % 10 == 0):
				print("[epoch "+str(epoch)+"]processing: "+str(100.0 * counter * self.batch_size / N)+"%" )
				sys.stdout.flush()
				a, b, c= self.sess.run([self.x_summary, self.gz_summary, self.gex_summary], feed_dict=d)
				self.summary_writer.add_summary(a, es)
				self.summary_writer.add_summary(b, es)
				self.summary_writer.add_summary(c, es)
			else:
				self.summary_writer.add_summary(zr, gs)
				self.summary_writer.add_summary(xr, es)
				self.summary_writer.add_summary(rds, es)
				self.summary_writer.add_summary(fds, es)
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

	def scale(self, x):
	    return x / 127.5 - 1

	def rescale(self, y):
		return (y + 1) * 127.5

	def encoder(self, image, reuse = False, scope_str = "Encoder"):
		with tf.variable_scope(scope_str) as scope:
			if(reuse):
				scope.reuse_variables()
			#input size 64 * 64 * 3
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

			#input size 32 * 32 * df_dim
			conv2 = tf.layers.conv2d(
				inputs = lrelu1, 
				filters = 2*self.df_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name = "conv2")
			bn2 = tf.layers.batch_normalization(
				inputs = conv2,
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
				name = "bn2")
			lrelu2 = lrelu(bn2, 0.2)

			#input size 16 * 16 * 2df_dim
			conv3 = tf.layers.conv2d(
				inputs = lrelu2, 
				filters = 4*self.df_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name = "conv3")
			bn3 = tf.layers.batch_normalization(
				inputs = conv3,
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
				name = "bn3")
			lrelu3 = lrelu(bn3, 0.2)

			#input size 8 * 8 * 4df_dim
			conv4 = tf.layers.conv2d(
				inputs = lrelu3, 
				filters = 8*self.df_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = False,
				name = "conv4")
			bn4 = tf.layers.batch_normalization(
				inputs = conv4,
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
				name = "bn4")
			lrelu4 = lrelu(bn4, 0.2)

			#input size 4 * 4 * 8df_dim
			conv5 = tf.layers.conv2d(
				inputs = lrelu4, 
				filters = self.z_dim, 
				kernel_size = 4, 
				strides = 2,
				padding = "same",
				kernel_initializer = tf.random_normal_initializer(stddev=0.02),
				use_bias = True,
				name = "conv5")

			#input size 2 * 2 * z_dim
			out = tf.layers.average_pooling2d(conv5, 2, 2)

			#input size 1 * 1 * z_dim
			out = tf.reshape(out, [-1, self.z_dim])
			out = tf.nn.l2_normalize(out, 1, name="out")
		return out


	def generator(self, z, reuse = False, scope_str = "Generator"):
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
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
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
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
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
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
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
				gamma_initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
				momentum=0.1,
				axis = -1,
				training = self.is_training,
				name="bn4")				
			relu4 = tf.nn.relu(bn4, name="relu4")

			#input size 32 * 32 * 2gf_dim
			deconv5 = tf.layers.conv2d_transpose(
					inputs = relu4,
					filters = self.c_dim, 
					kernel_size = 4,
					strides = 2,
					padding = "same",
					kernel_initializer = tf.random_normal_initializer(stddev=0.02),
					use_bias = False,
					name = "deconv5")

			#input size 64 * 64 * c_dim
			out = tf.tanh(deconv5, name="out")

		return out	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True,
	                    help='mnist | svhn | imagenet')
	parser.add_argument('--save_dir', default='None',
	                    help='folder to output model checkpoints')
	parser.add_argument('--log_dir', default='None', 
						help='folder to output tensorboard log')
	parser.add_argument('--save_every', default=5, type=int, help='')
	parser.add_argument('--restore', type=bool, default=False)


	parser.add_argument('--z_dim', type=int, default=128,
	                    help='size of the latent z vector, default is 128')
	parser.add_argument('--ngf', type=int, default=64)
	parser.add_argument('--ndf', type=int, default=64)
	parser.add_argument('--c_dim', type=int)

	parser.add_argument('--nepoch', type=int, default=25,
	                    help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.0002,
	                    help='learning rate, default=0.0002')
	parser.add_argument('--drop_lr', type=int, default=5, help='')
	parser.add_argument('--batch_size', type=int,
	                    default=64, help='batch size')

	parser.add_argument('--lamb', type=int, default=1000)
	parser.add_argument('--miu', type=int, default=10)
	parser.add_argument('--g_step', type=int, default=2, help='steps to train generater per encoder train step, default is 2')

	opt = parser.parse_args()		

	if(opt.dataset == "mnist"):
		print("load data...")
		mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
		X_train = np.reshape(mnist.train.images, [-1,28,28])
		X_val = np.reshape(mnist.validation.images, [-1,28,28])
		X_train = np.expand_dims(X_train, 3)
		X_val = np.expand_dims(X_val, 3)
		print("finish loading")
	elif(opt.dataset == "imagenet"):
		data_dir = "../data/data_imagenet"
		print("load data...")
		X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir, prefix="")
		print("finish loading")
	elif(opt.dataset == "celeba"):
		data_dir = "../data/data_celeba"
		print("load data...")
		X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir, prefix="")
		print("finish loading")
	elif(opt.dataset == "svhn"):
		data_dir = "../data/data_svhn"
		print("load data...")
		X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir, prefix="")
		print("finish loading")
	else:
		print('no such dataset!')
		return	
	if(opt.save_dir == 'None'):
		opt.save_dir = "../checkpoints/" + opt.dataset + "64.ckpt"
	if(opt.log_dir == 'None'):
		opt.log_dir = "../logs/" + opt.dataset + "64/"
	model = AGE_64(batch_size=opt.batch_size, lr=opt.lr, decay_every=opt.drop_lr,
		log_dir=opt.log_dir, save_dir=opt.save_dir, 
		c_dim=opt.c_dim, z_dim=opt.z_dim, 
		miu=opt.miu, lamb=opt.lamb, g_iter=opt.g_step)
	model.train(X_train, X_val, opt.nepoch, opt.restore)

if __name__ == "__main__":
	main()





















