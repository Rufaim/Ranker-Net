import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import axes3d

from layers import Dense, NALU

RANDOM_SEED = 42
NUM_CLASSES = 3
gen = np.random.RandomState(seed=RANDOM_SEED)

def build_nets():
	NET1 = [Dense(2,activation=tf.nn.elu,initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)),
			Dense(1,initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED))]

	NET2 = [NALU(2,initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)),
			Dense(1,initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED))]

	input_ph = tf.placeholder(tf.float32,[None,2])
	target_ph = tf.placeholder(tf.float32,[None,1])

	net_dense = input_ph
	net_nalu = input_ph
	for i,(l1,l2) in enumerate(zip(NET1,NET2)):
		net_dense = l1(net_dense,"dense_{}".format(i))
		net_nalu = l2(net_nalu,"nalu_{}".format(i))
	loss_dense = tf.reduce_mean(tf.abs(target_ph - net_dense))
	loss_nalu = tf.reduce_mean(tf.abs(target_ph - net_nalu))
	train_op_dense = tf.train.AdamOptimizer(0.001).minimize(loss_dense)
	train_op_nalu = tf.train.AdamOptimizer(0.001).minimize(loss_nalu)
	return input_ph, target_ph,\
		net_dense, net_nalu,\
		train_op_dense, train_op_nalu


def build_dataset():
	x = np.linspace(-5,5,1000)
	f = np.exp(-x**2/10)
	f_prime = -f*x/5.0
	return np.stack([f,x],axis=1),f_prime + gen.randn(x.shape[0])*0.05

X,Y = build_dataset()

input_ph, target_ph,net_dense, net_nalu, train_op_dense, train_op_nalu = build_nets()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100000):
		if i%100 ==0:
			print("Epoch #{}".format(i))
		idx = gen.randint(0,X.shape[0],size=80)

		sess.run([train_op_dense,train_op_nalu],feed_dict={input_ph:np.reshape(X[idx],[-1,2]),target_ph:np.reshape(Y[idx],[-1,1])})

	Y_dense,Y_nalu = sess.run([net_dense,net_nalu],feed_dict={input_ph:np.reshape(X,[-1,2])})


fig = pyplot.figure()
ax = axes3d.Axes3D(fig)
ax.plot3D(X[:, 0], X[:, 1], Y, "co",label="data")
ax.plot3D(X[:, 0], X[:, 1], np.reshape(Y_dense,[-1,]), "ro",label="Dense")
ax.plot3D(X[:, 0], X[:, 1], np.reshape(Y_nalu,[-1]), "bo",label="NALU")
pyplot.legend()
pyplot.show()