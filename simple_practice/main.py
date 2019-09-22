import tensorflow as tf
import os

from simple_practice.utils import Dataset, Model
# dataset
d = Dataset()
dataset_size = d.dataset_size
batch_size = d.batch_size
[X, Y] = d.getDataset()

# model 定义
m = Model()
x = m.x
y_gt = m.y_gt
steps = m.train_steps
lr = m.learning_rate
y = m.forward(x)

# loss 定义
mse_loss = tf.reduce_mean(tf.square(y_gt - y))
train_op = tf.train.AdamOptimizer(lr).minimize(mse_loss)

# tensorboard 日志信息
log_dir = 'log'
if os.path.exists(log_dir):
    os.system('rm -rf ' + log_dir)
os.system('mkdir ' + log_dir)

global_step = tf.contrib.framework.get_or_create_global_step()
tf.summary.scalar('loss', mse_loss)
tf.summary.scalar('global_step', global_step)
summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('train start!')

    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        summary_value, global_step_value, _ = sess.run([summary_op, global_step, train_op], feed_dict={x: X[start:end], y_gt: Y[start: end]})

        if i % 100 == 0:
            total_loss = sess.run(mse_loss, feed_dict={x:X, y_gt:Y})
            summary_writer.add_summary(summary_value, i)

    summary_writer.close()
    print('train finished!')


    print('test:')
    # x.shape: [b, 2]
    x_test = tf.expand_dims(tf.constant([2., 3.]), 0)
    y_test_gt = tf.constant([34.])
    y_predict = sess.run(y, feed_dict={x:x_test.eval()})
    print('expect value:{0}, predict value:{1}'.format(y_test_gt.eval().item(), y_predict.item()))

    # save model
    print('save model')
    saver = tf.train.Saver()
    model_dir = 'saved_model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    saver.save(sess, 'saved_model/model_practice.ckpt')
    print('save finish!')