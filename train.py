from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import numpy as np
import abstractions.file_storage as file_storage
import os
from tcn import TCN
import argparse
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser(description='skynet brain nn')
    parser.add_argument('--model-idx',
                        type=int,
                        help='model index')
    parser.add_argument('--model-dir',
                        type=str,
                        help='path to save the model')
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='input batch size for training')
    parser.add_argument('--test-split',
                        type=float,
                        default=0.2,
                        help='split size for training / testing dataset')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='number of epochs to train')
    parser.add_argument('--dropout-rate',
                        type=float,
                        default=0.5,
                        help='dropout rate')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed')
    args = parser.parse_args()
    return args

args = get_args()

ds_remote_path = f"batches/{args.model_idx}.npy"
ds_local_path = f"input/ds_{args.model_idx}.npy"

if not os.path.exists(ds_local_path):
    file_storage.get_file(ds_remote_path, ds_local_path)

#load data
arr = np.load(ds_local_path)
x = arr[:, :, 0:10]
y = arr[:, 99, 10]

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

# tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
#     FLAGS.tpu,
#     zone=FLAGS.tpu_zone,
#     project=FLAGS.gcp_project)
#
# config = tpu_config.RunConfig(
#     cluster=tpu_cluster_resolver,
#     model_dir=FLAGS.model_dir,
#     save_checkpoints_steps=max(600, FLAGS.iterations_per_loop),
#     tpu_config=tpu_config.TPUConfig(
#         iterations_per_loop=FLAGS.iterations_per_loop,
#         num_shards=FLAGS.num_cores,
#         per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

with tpu_strategy.scope():
    # set keras model
    i = Input(batch_shape=(None, 100, 10))

    o = TCN(return_sequences=False, dropout_rate=args.dropout_rate, dilations=(1, 2, 4, 8, 16, 32, 64))(i)  # The TCN layers are here.
    o = Dense(1)(o)

    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='mse')

    # train
    m.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.test_split)

    # save model
    local_model_path = f"output/{args.model_idx}_{args.epochs}.h5"
    file_storage.create_dir_if_absent(local_model_path)
    remote_model_path = f"models/{args.model_idx}_{args.epochs}.h5"
    m.save(local_model_path, save_format='h5')
file_storage.put_file(local_model_path, remote_model_path)
