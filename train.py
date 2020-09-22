import os
import sys
from os.path import join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import nn
import data_loader
import config

strategy = tf.distribute.MirroredStrategy()
nb_gpu = strategy.num_replicas_in_sync
global_batch = nb_gpu * config.batch_size

dataset = data_loader.input_fn()
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(5e-4)
    output_dim = len(data_loader.TextFeaturizer(config.vocabulary_path).token_to_index)
    model = nn.build_model(config.input_dim, output_dim)
    model.summary()

with strategy.scope():
    loss_object = nn.get_loss()


    def compute_loss(_labels, _logits):
        per_example_s_loss = loss_object(_labels, _logits)
        scale_loss = tf.reduce_sum(per_example_s_loss) * 1. / global_batch

        return scale_loss

with strategy.scope():
    def train_step(_inputs, _labels):
        with tf.GradientTape() as tape:
            loss_value = compute_loss(_labels, model(_inputs))
        train_variable = model.trainable_variables
        gradient = tape.gradient(loss_value, train_variable)
        optimizer.apply_gradients(zip(gradient, train_variable))

        return loss_value

with strategy.scope():
    @tf.function
    def distribute_train_step(_inputs, _labels):
        loss_value = strategy.run(train_step, args=(_inputs, _labels))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_value, axis=None)

if __name__ == '__main__':
    nb_steps = len(data_loader.preprocess_data(config.train_path)) // global_batch
    print(f"--- Start Training with {nb_steps} Steps ---")
    for step, inputs in enumerate(dataset):
        step += 1
        loss = distribute_train_step(inputs[0], inputs[1])
        print('---', step, '---', loss.numpy())
        if step % nb_steps == 0:
            model.save_weights(join("weights", "model.h5"))
        if step // nb_steps == config.nb_epochs:
            sys.exit("--- Stop Training ---")
