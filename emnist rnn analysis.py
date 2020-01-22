from keras.models import Model
from keras.layers import LSTM, GRU, Input,GlobalMaxPool1D, Dropout, Dense
from keras.optimizers import Adagrad,Adam
import pandas as pd
import numpy as np
import gzip

import keras.backend as k
if len(k.tensorflow_backend._get_available_gpus())>0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU


data=pd.read_csv('train_dataM.csv')
with gzip.open('../emnist-gzip/gzip/emnist-byclass-train-labels-idx1-ubyte.gz', 'r') as f:
    train_labels=f.read()

train_labels= np.frombuffer(train_labels, dtype=np.uint8).astype(np.float32)
train_labels=train_labels[8:]

VALIDATION_SPLIT=0.2
EPOCHs = 10
BATCH_SIZE=150

input_ = Input(shape=(787,))

x = LSTM(150, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
output = Dense(62, activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
  loss='sparse_binary_crossentropy',
  optimizer=Adam(lr=0.01),
  metrics=['accuracy']
)

print('Training model...')
r = model.fit(
  data,
  targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)


folder='images'
Names = ["acc", "loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
titles=['Accuracies', 'Losses']
# loop over the accuracy names
for (i, l) in enumerate(Names):
    # plot the loss for both the training and validation data
    ax[i].set_title(titles[i])
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel(titles[i])
    ax[i].plot(np.arange(0, Epochs), r.history[l], label='train_'+l)
    ax[i].plot(np.arange(0, Epochs), r.history["val_" + l],
        label="val_" + l)
    ax[i].legend()

plt.figure(figsize=(12,20))
plt.show()
# save the accuracies figure
plt.tight_layout()
fig.savefig(f"{folder}/RNN_accs.png")
plt.close()
