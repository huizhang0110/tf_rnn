import tensorflow as tf
import numpy as np


def my_dynamic_rnn(cell, sequence_length, inputs, time_major=True):
    if time_major is False:
        inputs = tf.transpose(inputs, [1, 0, 2])
    max_time = tf.shape(inputs)[0]
    batch_size = tf.shape(inputs)[1]
    input_depth = tf.shape(inputs)[2]

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(inputs)

    def loop_fn_initial():
        initial_elements_finished = (0 >= sequence_length)
        initial_input = inputs_ta.read(0)
        initial_cell_state = cell.zero_state(batch_size, tf.float32)
        initial_cell_output = None
        initial_loop_state = None
        return (initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state)

    def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

        def get_net_input():
          #   output_logits = tf.add(tf.matmul(previous, W), b)
            # prediction = tf.argmax(output_logits, axis=1)
          #   next_input = tf.nn.embedding_lookup(embedding, prediction)
            next_input = inputs_ta.read(time)
            return next_input
        
        elements_finished = (time >= sequence_length)
        finished = tf.reduce_all(elements_finished)
        next_input = tf.cond(
            finished, 
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: get_net_input(),
        )
        state = previous_state
        output = previous_output
        loop_state = None
        return (elements_finished, next_input, state, output, loop_state)

    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:
            assert previous_output is None and previous_state is None
            return loop_fn_initial()
        else:
            return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    output_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = output_ta.stack()
    return outputs, final_state


X = np.array([
    [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8]
    ],
    [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ]
]).astype(np.float32)
X_lengths = [4, 3]  # 输入每个Example的长度，调用tf.nn.dynamic_rnn的时候，传递sequence_lengths参数，可以自动处理变长的情况

input_x_placeholder = tf.placeholder(shape=[None, 4, 3], dtype=tf.float32)
sequence_length_placeholder = tf.placeholder(shape=[None, ], dtype=tf.int32)

cell = tf.nn.rnn_cell.LSTMCell(num_units=2)

# rnn_outputs, last_states = tf.nn.dynamic_rnn(
    # cell=cell,
    # sequence_length=X_lengths,
    # inputs=X,
    # dtype=tf.float32
# )
rnn_outputs, last_states = my_dynamic_rnn(
    cell=cell,
    sequence_length=sequence_length_placeholder,
    inputs=input_x_placeholder
)

print("rnn_outputs: ", rnn_outputs)
print("last_states: ", last_states)


with tf.Session() as sess:
    sess.run([
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ])
    rnn_outputs_ = sess.run(rnn_outputs)

    print(rnn_outputs_[0])
    print(rnn_outputs_[1])

