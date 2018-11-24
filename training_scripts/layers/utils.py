import tensorflow as tf

def _activation_to_string(activation):
    res = ''
    if activation is tf.nn.relu:
        res = 'R'
    if activation is tf.identity:
        res = 'I'
    if activation is tf.nn.sigmoid:
    	res = 'S'
    return res