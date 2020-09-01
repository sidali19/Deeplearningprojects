import numpy as np
from keras.models import model_from_yaml


def saveModel(model, savename):
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
        # serialize weights to HDF5
        model.save_weights(savename + ".h5")
    print("Weights", savename, ".h5 saved to disk")


maxLCap = 35
nbkeep = 1000

outfile = 'Training_data_' + str(nbkeep) + '.npz'
data = np.load(outfile)
X_train = data['X_train']
y_train = data['Y_train']

from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking
from keras.optimizers import Adam

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(maxLCap, 202)))
model.add(SimpleRNN(100, return_sequences=True, input_shape=(nbkeep, maxLCap), unroll=True))
model.add(Dense(nbkeep))
model.add(Activation("softmax"))

BATCH_SIZE = 10
NUM_EPOCHS = 10

adam = Adam()
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
scores_train = model.evaluate(X_train, y_train, verbose=1)
print("PERFS TRAIN: %s: %.2f%%" % (model.metrics_names[1], scores_train[1] * 100))

"""
PERFS TRAIN: acc: 45.13%
"""

saveModel(model, savename='model-tp5')