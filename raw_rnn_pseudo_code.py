import tensorflow as tf


time = tf.constant(0, dtype=tf.int32)
(finished, next_input, initial_state, emit_structure, loop_state) = loop_fn(
        time=time, cell_output=None, cell_state=None, loop_state=None)
emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
state = initial_state

while not all(finished):
    (output, cell_state) = cell(next_input, state)
    (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
            time=time + 1, cell_output=output, cell_state=cell_state,
            loop_state=loop_state)
    # Emit zeros and copy forward state for minibatch entries that are finished.
    state = tf.where(finished, state, next_state)
    emit = tf.where(finished, tf.zero_like(emit_structure), emit)
    emit_ta = emit_ta.write(time, emit)
    # If any new minibatch entries are marked finished, mark these
    finished = tf.logical_or(finished, next_finished)
    time += 1

return (emit_ta, state, loop_state)


# A simple implementation of dynamic_rnn via raw_rnn looks like this.
inputs = tf.placeholder(shape=[max_time, batch_size, input_depth], dtype=tf.float32)
sequence_length = tf.placeholder(shape=[batch_size,], dtype=tf.int32)
inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
inputs_ta = inputs_ta.unstack(inputs)
cell = tf.contrib.rnn.LSTMCell(num_units)

def loop_fn(time, cell_output, cell_state, loop_state):
    emit_output = cell_output
    if cell_output is None:
        next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
        next_cell_state = cell_state

    elements_finished = (time >= sequence_length)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        lambda: inputs_ta.read(time)
    )
    next_loop_state = None
    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

output_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
outputs = output_ta.stack()

"""Tensorflow official documentation

"""




"""Note from https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/

What is the initial state or the input to the cell?              if cell_output is None
What is the next state or the next input to the cell?            else branch
What information do you want to propagate through the network?   loop_state.write 
When will the recurrence stop?                                   elements_finished

output_ta = tf.TensorArray(size=784, dtype=tf.float32)

def loop_fn(time, cell_output, cell_state, loop_state):
    emit_output = cell_output

    if cell_output is None:
        next_cell_state = cell_init_state
        next_pixel = cell_init_pixel
        next_loop_state = output_ta
    else:
        next_cell_state = cell_state
        next_pixel = tf.cond(is_training,
            lambda: inputs_ta.read(time - 1),
            lambda: tf.contrib.distribution.Bernoulli(
                probs=tf.nn.sigmoid(
                    tf.layers.dense(cell_output, 1, name="output_to_p", 
                    activation=tf.nn.sigmoid, reuse=True)
                ),
                dtype=tf.float32
            )
        )
        next_loop_state = loop_state.write(time - 1, next_pixel)

    elements_finished = time >= 784
    
    return (elements_finished, next_pixel, next_cell_state, emit_output, next_loop_state)
"""

