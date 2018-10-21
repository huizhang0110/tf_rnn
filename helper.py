import numpy as np


def batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # PAD 0
    for i, seq in enumerate(inputs):
        inputs_batch_major[i, :len(seq)] = seq

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths


def random_sequence(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    assert length_from <= length_to
    def random_length():
        return np.random.randint(length_from, length_to + 1)
    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)
        ]


def test_helper():
    batch_size = 100
    batches = random_sequence(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)
    for seq in next(batches)[:10]:
        print(seq)


if __name__ == "__main__":
    test_helper()
