from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import numpy as np
import abstractions.file_storage as file_storage
import os
from tcn import TCN
import argparse
import tensorflow as tf
import timeit

def get_args():
    def boolean_string(s: str):
        s = s.lower()
        if s not in {'false', 'true', 'y', 'n'}:
            raise ValueError('Not a valid boolean string')
        return s == 'true' or s == 'y'

    parser = argparse.ArgumentParser(description='skynet brain nn')
    parser.add_argument('--model-idx',
                        type=int,
                        help='model index')
    parser.add_argument('--model-dir',
                        type=str,
                        help='path to save the model')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='input batch size for training')
    parser.add_argument('--test-split',
                        type=float,
                        default=0.2,
                        help='split size for training / testing dataset')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train')
    parser.add_argument('--dropout-rate',
                        type=float,
                        default=0.5,
                        help='dropout rate')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed')
    parser.add_argument('--use-tpu',
                        default = False,
                        type=boolean_string,
                        help = 'use tpu')
    args, _ = parser.parse_known_args()
    return args


def load_data(args):
    ds_remote_path = f"batches/{args.model_idx}.npy"
    ds_local_path = f"input/ds_{args.model_idx}.npy"

    if not os.path.exists(ds_local_path):
        file_storage.get_file(ds_remote_path, ds_local_path)

    # load data
    arr = np.load(ds_local_path)
    x = arr[:, :, 0:10]
    y = arr[:, 99, 10]
    return x, y

    # total_samples = x.shape[0]
    # num_batches = total_samples // args.batch_size
    # num_samples = num_batches * args.batch_size
    #
    # return x[0:num_samples, :, :], y[0: num_samples]


def build_model():
    input = Input(batch_shape=(None, 100, 10))

    output = TCN(return_sequences=False, dropout_rate=args.dropout_rate, dilations=(1, 2, 4, 8, 16, 32, 64))(
        input)  # The TCN layers are here.
    output = Dense(1)(output)

    model = Model(inputs=[input], outputs=[output])
    return model


def save_model(model):
    local_model_path = f"output/{args.model_idx}_{args.epochs}.h5"
    file_storage.create_dir_if_absent(local_model_path)
    remote_model_path = f"models/{args.model_idx}_{args.epochs}.h5"
    model.save(local_model_path, save_format='h5')
    file_storage.put_file(local_model_path, remote_model_path)


def get_distibution_strategy(args):
    if args.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.experimental.TPUStrategy(resolver)
    else:
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")


args = get_args()
x, y = load_data(args)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(args.batch_size, drop_remainder=True)
strategy = get_distibution_strategy(args)
print(f"Num replicas: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_model()
    # print(model.summary())
    model.compile(optimizer='adam', loss='mse')
    start_time = timeit.default_timer()
    model.fit(dataset, epochs=args.epochs)
    # model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size)
    print(timeit.default_timer() - start_time)
    save_model(model)
