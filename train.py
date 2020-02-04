from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import numpy as np

from tcn import TCN

batch_size, timesteps, input_dim = None, 100, 10

arr = np.load("batches_1.npy")
x = arr[:,:,0:10]
y = arr[:,99,10]

i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=False, dropout_rate=0.5, dilations=(1, 2, 4, 8, 16, 32, 64))(i1)  # The TCN layers are here.
o = Dense(1)(o)

m1 = Model(inputs=[i], outputs=[o])
m1.compile(optimizer='adam', loss='mse')

m1.fit(x, y, epochs=10, batch_size=100, validation_split=0.2)

