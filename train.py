from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import numpy as np

from tcn import TCN

batch_size, timesteps, input_dim = None, 100, 10

arr = np.load("batches_1.npy")
x = arr[:,:,0:10]
y = arr[:,99,10]
# y = arr[:,99,20]

# i = Input(batch_shape=(batch_size, timesteps, input_dim))
#
# o = TCN(return_sequences=False, dropout_rate=0.0, dilations=(1, 2, 4, 8, 16, 32, 64))(i)  # The TCN layers are here.
# o = Dense(1)(o)
#
# m = Model(inputs=[i], outputs=[o])
# m.compile(optimizer='adam', loss='mse')
#
# m.fit(x, y, epochs=1000, batch_size=100, validation_split=0.2)

i1 = Input(batch_shape=(batch_size, timesteps, input_dim))

o1 = TCN(return_sequences=False, dropout_rate=0.5, dilations=(1, 2, 4, 8, 16, 32, 64))(i1)  # The TCN layers are here.
o1 = Dense(1)(o1)

m1 = Model(inputs=[i1], outputs=[o1])
m1.compile(optimizer='adam', loss='mse')

m1.fit(x, y, epochs=30000, batch_size=100, validation_split=0.2)

