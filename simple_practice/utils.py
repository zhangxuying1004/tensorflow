import tensorflow as tf
from tensorflow import nn
from numpy.random import RandomState

class Dataset:
    def __init__(self):
        self.dataset_size = 128
        self.batch_size = 8
        self.rdm = RandomState()

    def getDataset(self):
        X = self.rdm.rand(self.dataset_size, 2)
        Y = [[5 * x1 + 8 * x2] for (x1, x2) in X]
        return [X, Y]

class Model:
    def __init__(self):
        # 参数设置
        self.w1 = tf.Variable(tf.random_normal([2, 10], stddev=1, seed=1))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[10]))
        self.w2 = tf.Variable(tf.random_normal([10, 10], stddev=1, seed=1))
        self.b2 = tf.Variable(tf.constant(0.1, shape=[10]))
        self.w3 = tf.Variable(tf.random_normal([10, 1], stddev=1, seed=1))
        self.b3 = tf.Variable(tf.constant(0.1, shape=[1]))

        # 占位符设置
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
        self.y_gt = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # 参数设置
        self.train_steps = 10000
        self.learning_rate = 1e-2

    def forward(self, x):
        l1 = nn.relu(tf.matmul(x, self.w1) + self.b1)
        l2 = nn.relu(tf.matmul(l1, self.w2) + self.b2)
        y = tf.matmul(l2, self.w3) + self.b3
        return y
