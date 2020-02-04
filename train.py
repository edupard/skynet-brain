from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import numpy as np
import abstractions.file_storage as file_storage
import os

from tcn import TCN

batch_size, timesteps, input_dim = None, 100, 10

ds_remote_path = "batches/1.npy"
ds_local_path = "data/data.npy"

if not os.path.exists(ds_local_path):
    file_storage.get_file(ds_remote_path, ds_local_path)

arr = np.load(ds_local_path)
x = arr[:,:,0:10]
y = arr[:,99,10]

i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=False, dropout_rate=0.5, dilations=(1, 2, 4, 8, 16, 32, 64))(i1)  # The TCN layers are here.
o = Dense(1)(o)

m1 = Model(inputs=[i], outputs=[o])
m1.compile(optimizer='adam', loss='mse')

m1.fit(x, y, epochs=10, batch_size=100, validation_split=0.2)

