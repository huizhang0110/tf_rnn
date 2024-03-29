import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import helper


PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2

encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name="encoder_inputs")
encoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name="encoder_inputs_length")
decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets")

# Embeddings
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
print("encoder_inputs_embedded: ", encoder_inputs_embedded)

# Encoder
encoder_cell = LSTMCell(encoder_hidden_units)
((encoder_fw_outputs, encoder_bw_outputs), \
        (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=encoder_cell,
    cell_bw=encoder_cell,
    inputs=encoder_inputs_embedded,
    sequence_length=encoder_inputs_length,
    dtype=tf.float32,
    time_major=True
)
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)
# Decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3
# +2 addition steps, +1 leading <EOS> token for decoder inputs

# Output projection
# Decoder will contain manually specified by us transition step
# output(t) -> output projction(t) -> prediction(t) (argmax) -> input embedding(t+1) -> input(t+1)
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

# Decoder via tf.nn.raw_rnn
assert EOS == 1 and PAD == 0
eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name="EOS")
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name="PAD")
eos_step_embeded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embeded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # All false at the initial step
    initial_input = eos_step_embeded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # We don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished, 
        lambda: pad_step_embeded, 
        get_next_input
    )
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            next_input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


decoder_output_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_output_ta.stack()
print("decoder_outputs: ", decoder_outputs)

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, [-1, decoder_dim])
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, [decoder_max_steps, decoder_batch_size, vocab_size])
decoder_prediction = tf.argmax(decoder_logits, axis=-1)

# Optimizer
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

batch_size = 100
batches = helper.random_sequence(length_from=3, length_to=9, vocab_lower=2, vocab_upper=10, batch_size=batch_size) 

def next_feed():
    batch = next(batches)
    encoder_inputs_, encoder_inputs_length_ = helper.batch(batch)
    decoder_targets_, _ = helper.batch(
        [seq + [EOS] + [PAD] * 2 for seq in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_inputs_length_,
        decoder_targets: decoder_targets_
    }


max_batches = 3001
batches_in_epoch = 100

loss_track = []
with tf.Session() as sess:
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ])
    # Training on the toy task
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch % batches_in_epoch == 0:
            print("batch {}".format(batch))
            print("  minibatch loss: {}".format(l))
            prediction_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, prediction_.T)):
                print("  sample {}".format(i))
                print("  input > {}".format(inp))
                print("  pred  > {}".format(pred))
                if i >= 2:
                    break
            print()
        


