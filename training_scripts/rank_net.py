import numpy as np
import tensorflow as tf
import json

from layers import Dense

class RankNetwork(object):
    def __init__(self,input_len, net_structure, learning_rate=1e-4,alphas=None):
        self.input_len = input_len
        self.net_structure = net_structure
        self.net_structure.append(Dense(1)) #,tf.nn.relu

        self.learning_rate = learning_rate
        if alphas is None or len(alphas)!=2:
            self.alphas = [1,1]
        else:
            self.alphas = alphas
        self.params = []

        self._build_placholders()
        self._build_net()
        self._build_output()
        self._build_loss()
        self._build_optimizer()

        self.init = tf.global_variables_initializer()

    def _build_placholders(self):
        self.input_1 = tf.placeholder(tf.float32,[None,self.input_len])
        self.input_2 = tf.placeholder(tf.float32,[None,self.input_len])
        self.learning_rate_ph = tf.placeholder(tf.float32,[])

    def _build_net(self):
        net1 = self.input_1
        net2 = self.input_2

        for i,layer in enumerate(self.net_structure):
            layer_name = "layer"+str(i)
            with tf.variable_scope(layer_name):
                net1 = layer(net1)
                net2 = layer(net2)

        self.net_out_1, self.net_out_2 = net1, net2

    def _build_output(self):
        self.predicts = self.net_out_1

    def _build_loss(self):
        loss = -tf.log_sigmoid(self.net_out_1 - self.net_out_2)

        loss1 = -tf.log_sigmoid(self.net_out_1 - 100.0)
        loss2 = -tf.log_sigmoid(-100-self.net_out_2)

        self.loss = tf.reduce_sum(loss) + tf.constant(1/self.alphas[0])*tf.reduce_sum(loss1) + \
                    tf.constant(1/self.alphas[1])*tf.reduce_sum(loss2) + \
                     + 0.1*sum(tf.reduce_sum(tf.abs(p[0])) for p,_,_ in self.params)
        self.loss_median = tf.contrib.distributions.percentile(loss, 50)
        self.accuracy = tf.reduce_mean(tf.cast(tf.greater(self.net_out_1,self.net_out_2), tf.float32))

    def _build_optimizer(self):
        self.global_step = tf.Variable(0,trainable=False)
        self.learning_rate = tf.Variable(self.learning_rate,trainable=False)
        self.assign_LR = self.learning_rate.assign(self.learning_rate_ph)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)

    def initialize(self,sess):
        self.sess = sess
        self.sess.run(self.init)

    def predict(self,X1):
        return self.sess.run(self.predicts, feed_dict={
                        self.input_1: np.reshape(X1,(-1,self.input_len))
                        })

    def train_step(self,X1,X2):
        self.sess.run(self.opt,feed_dict={
                    self.input_1: np.reshape(X1,(-1,self.input_len)),
                    self.input_2: np.reshape(X2,(-1,self.input_len))
                    })
    def get_metrics(self,X1,X2):
        return  self.sess.run([self.loss,self.accuracy, self.loss_median],feed_dict={
                        self.input_1: np.reshape(X1,(-1,self.input_len)),
                        self.input_2: np.reshape(X2,(-1,self.input_len))
                        })
    def get_global_step(self):
        return self.sess.run(self.global_step)

    def get_learning_rate(self):
        return self.sess.run(self.learning_rate)

    def set_learning_rate(self,new_LR):
        return self.sess.run(self.assign_LR,feed_dict={self.learning_rate_ph: new_LR})

    def dump_network_to_file(self,filename):
        print('Dumping network to file ' + filename)
        res = {}
        for i,layer in enumerate(self.net_structure):
            layer_name = "layer"+str(i)
            curr_layer = layer.to_json(self.sess)
            res[layer_name] = curr_layer
        res["parameters"] = {"use_abs":True}
        with open(filename, 'w') as f:
            json.dump(res, f)
   
