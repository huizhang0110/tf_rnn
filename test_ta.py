import tensorflow as tf


def condiction(time, output_ta_l):
    return tf.less(time, 3)


def body(time, output_ta_l):
    output_ta_l = output_ta_l.write(time, [2.4, 3.5])
    return time + 1, output_ta_l


time = tf.constant(0)
output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
result = tf.while_loop(condiction, body, loop_vars=[time, output_ta])

last_time, last_out = result
final_out = last_out.stack()

with tf.Session() as sess:
    sess.run(last_time)
    sess.run(final_out)
    # sess.run(last_out) Can not convert a TensorArray into a Tensor or Operation, You need to .stack to make it a Tensor
