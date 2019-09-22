import tensorflow as tf
import os
from simple_practice.utils import Model

m = Model()
x = m.x
y = m.forward(x)

with tf.Session() as sess:
    saver = tf.train.Saver()
    model_dir = 'saved_model/'
    if os.path.exists(model_dir):
        saver.restore(sess, model_dir + 'model_practice.ckpt')
        print('test:')
        # x.shape: [b, 2]
        x_test = tf.expand_dims(tf.constant([2., 3.]), 0)
        y_test_gt = tf.constant([34.])

        y_predict = sess.run(y, feed_dict={x:x_test.eval()})
        print('expect value:{0}, predict value:{1}'.format(y_test_gt.eval().item(), y_predict.item()))

    else:
        print('model does not exist')