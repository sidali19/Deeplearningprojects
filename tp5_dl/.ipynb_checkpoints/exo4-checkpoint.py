from keras.models import model_from_yaml
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking
import matplotlib.image as mpimg
import pandas as pd
import _pickle as pickle
from keras.models import model_from_yaml
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

maxLCap = 35
nbkeep = 1000


def loadModel(savename):
    with open(savename + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ", savename, ".yaml loaded ")
    model.load_weights(savename + ".h5")
    print("Weights ", savename, ".h5 loaded ")
    return model


def sampling(preds, temperature=0.1):
    preds = np.asarray(preds).astype('float64')
    predsN = pow(preds, 1.0 / temperature)
    predsN /= np.sum(predsN)
    probas = np.random.multinomial(1, predsN, 1)
    return np.argmax(probas)


# LOADING MODEL
nameModel = 'model-tp5'
model = loadModel(nameModel)

optim = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])

# LOADING TEST DATA
outfile = 'Testing_data_' + str(nbkeep) + '.npz'
npzfile = np.load(outfile)

X_test = npzfile['X_test']
Y_test = npzfile['Y_test']

outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
[listwords, embeddings] = pickle.load(open(outfile, "rb"))
indexwords = {}
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

ind = np.random.randint(X_test.shape[0])

filename = 'flickr_8k_test_dataset.txt'  # PATH IF NEEDED

df = pd.read_csv(filename, delimiter='\t')
iter = df.iterrows()

for i in range(ind + 1):
    x = iter.__next__()

imname = x[1][0]
print("image name=" + imname + " caption=" + x[1][1])
dirIm = "96420612_feb18fc6c6.jpg"  # CHANGE WITH YOUR DATASET

img = mpimg.imread(dirIm + imname)
plt.figure(dpi=100)
plt.imshow(img)
plt.axis('off')
plt.show()

pred = model.predict(X_test[ind:ind + 1, :, :])

nbGen = 5
temperature = 0.1  # Temperature param for peacking soft-max distribution

for s in range(nbGen):
    wordpreds = "Caption n° " + str(s + 1) + ": "
    indpred = sampling(pred[0, 0, :], temperature)
    wordpred = listwords[indpred]
    wordpreds += str(wordpred) + " "
    X_test[ind:ind + 1, 1, 0:102] = embeddings[listwords.index(wordpred)]  # COMPLETE WITH YOUR CODE
    cpt = 1
    while (str(wordpred) != '<end>' and cpt < 30):
        pred = model.predict(X_test[ind:ind + 1, :, :])
        indpred = sampling(pred[0, cpt, :], temperature)
        wordpred = listwords[indpred]
        wordpreds += str(wordpred) + " "
        cpt += 1
        X_test[ind:ind + 1, cpt, 0:102] = embeddings[listwords.index(wordpred)]  # COMPLETE WITH YOUR CODE

    print(wordpreds)

# BLUE


from keras.optimizers import RMSprop, Adam
from keras.models import model_from_yaml
import pandas as pd
import numpy as np
import nltk

# LOADING TEST DATA
nbkeep = 1000
outfile = 'Testing_data_' + str(nbkeep) + '.npz'
npzfile = np.load(outfile)

X_test = npzfile['X_test']
Y_test = npzfile['Y_test']

# LOADING MODEL
model = loadModel(nameModel)

# COMPILING MODEL
optim = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])
scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1] * 100))

# LOADING TEXT EMBEDDINGS
outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
[listwords, embeddings] = pickle.load(open(outfile, "rb"))
indexwords = {}
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

# COMPUTING CAPTION PREDICTIONS ON TEST SET
predictions = []
nbTest = X_test.shape[0]
for i in range(0, nbTest, 5):
    pred = model.predict(X_test[i:i + 1, :, :])
    wordpreds = []
    indpred = np.argmax(pred[0, 0, :])
    wordpred = listwords[indpred]
    wordpreds.append(str(wordpred))
    X_test[i, 1, 0:102] = embeddings[indpred]
    cpt = 1
    while (str(wordpred) != '<end>' and cpt < (X_test.shape[1] - 1)):
        pred = model.predict(X_test[i:i + 1, :, :])
        indpred = np.argmax(pred[0, cpt, :])
        wordpred = listwords[indpred]
        if (wordpred != '<end>'):
            wordpreds.append(str(wordpred))
        cpt += 1
        X_test[i, cpt, 0:102] = embeddings[indpred]

    if (i % 1000 == 0):
        print("i=" + str(i) + " " + str(wordpreds))
    predictions.append(wordpreds)

# LOADING GROUD TRUTH CAPTIONS ON TEST SET
references = []
filename = 'flickr_8k_test_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
iter = df.iterrows()

ccpt = 0
for i in range(int(nbTest / 5)):
    captions_image = []
    for j in range(5):
        x = iter.__next__()
        ll = x[1][1].split()
        caption = []
        for k in range(1, len(ll) - 1):
            caption.append(ll[k])

        captions_image.append(caption)
        ccpt += 1

    references.append(captions_image)

# COMPUTING BLUE-1, BLUE-2, BLUE-3, BLUE-4
blue_scores = np.zeros(4)
weights = np.zeros((4, 4))
weights[0, 0] = 1
weights[1, 0] = 0.5
weights[1, 1] = 0.5
weights[2, 0] = 1.0 / 3.0
weights[2, 1] = 1.0 / 3.0
weights[2, 2] = 1.0 / 3.0
weights[3, :] = 1.0 / 4.0

for i in range(4):
    blue_scores[i] = nltk.translate.bleu_score.corpus_bleu(references, predictions, weights=(
        weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3]))
    print("blue_score - " + str(i) + "=" + str(blue_scores[i]))


"""
output: 

Weights  model-tp5 .h5 loaded 
image name=3123463486_f5b36a3624.jpg caption=<start> The brown , white and black dog runs on a gravel surface . <end>
Caption n° 1: a dog is running through the grass . <end> 
Caption n° 2: a dog is running through the grass . <end> 
Caption n° 3: a dog is running through the grass . <end> 
Caption n° 4: a dog is running through the grass . <end> 
Caption n° 5: a dog is running through the grass . <end> 
Yaml Model  model-tp5 .yaml loaded 
Weights  model-tp5 .h5 loaded 

  32/5000 [..............................] - ETA: 1:00
 128/5000 [..............................] - ETA: 17s 
 224/5000 [>.............................] - ETA: 10s
 320/5000 [>.............................] - ETA: 8s 
 416/5000 [=>............................] - ETA: 6s
 512/5000 [==>...........................] - ETA: 5s
 608/5000 [==>...........................] - ETA: 5s
 704/5000 [===>..........................] - ETA: 4s
 800/5000 [===>..........................] - ETA: 4s
 896/5000 [====>.........................] - ETA: 4s
 992/5000 [====>.........................] - ETA: 4s
1088/5000 [=====>........................] - ETA: 3s
1152/5000 [=====>........................] - ETA: 3s
1216/5000 [======>.......................] - ETA: 3s
1312/5000 [======>.......................] - ETA: 3s
1408/5000 [=======>......................] - ETA: 3s
1504/5000 [========>.....................] - ETA: 3s
1600/5000 [========>.....................] - ETA: 3s
1696/5000 [=========>....................] - ETA: 2s
1792/5000 [=========>....................] - ETA: 2s
1888/5000 [==========>...................] - ETA: 2s
1984/5000 [==========>...................] - ETA: 2s
2080/5000 [===========>..................] - ETA: 2s
2176/5000 [============>.................] - ETA: 2s
2272/5000 [============>.................] - ETA: 2s
2368/5000 [=============>................] - ETA: 2s
2464/5000 [=============>................] - ETA: 2s
2560/5000 [==============>...............] - ETA: 1s
2656/5000 [==============>...............] - ETA: 1s
2752/5000 [===============>..............] - ETA: 1s
2848/5000 [================>.............] - ETA: 1s
2944/5000 [================>.............] - ETA: 1s
3040/5000 [=================>............] - ETA: 1s
3136/5000 [=================>............] - ETA: 1s
3232/5000 [==================>...........] - ETA: 1s
3328/5000 [==================>...........] - ETA: 1s
3424/5000 [===================>..........] - ETA: 1s
3520/5000 [====================>.........] - ETA: 1s
3616/5000 [====================>.........] - ETA: 1s
3712/5000 [=====================>........] - ETA: 0s
3808/5000 [=====================>........] - ETA: 0s
3872/5000 [======================>.......] - ETA: 0s
3968/5000 [======================>.......] - ETA: 0s
4032/5000 [=======================>......] - ETA: 0s
4096/5000 [=======================>......] - ETA: 0s
4160/5000 [=======================>......] - ETA: 0s
4224/5000 [========================>.....] - ETA: 0s
4288/5000 [========================>.....] - ETA: 0s
4384/5000 [=========================>....] - ETA: 0s
4480/5000 [=========================>....] - ETA: 0s
4576/5000 [==========================>...] - ETA: 0s
4672/5000 [===========================>..] - ETA: 0s
4768/5000 [===========================>..] - ETA: 0s
4864/5000 [============================>.] - ETA: 0s
4992/5000 [============================>.] - ETA: 0s
5000/5000 [==============================] - 4s 756us/step
PERFS TEST: acc: 42.85%
i=0 ['a', 'brown', 'dog', 'is', 'running', 'through', 'the', 'snow', '.']
i=1000 ['a', 'young', 'girl', 'in', 'a', 'red', 'shirt', 'is', 'sitting', 'on', 'a', 'bench', '.']
i=2000 ['a', 'man', 'in', 'a', 'blue', 'shirt', 'and', 'a', 'blue', 'shirt', 'and', 'a', 'black', 'shirt', 'is', 'standing', 'on', 'a', 'rock', '.']
i=3000 ['a', 'man', 'in', 'a', 'black', 'shirt', 'and', 'black', 'pants', 'and', 'black', 'pants', 'and', 'black', 'shorts', 'and', 'a', 'black', 'shirt', 'is', 'standing', 'on', 'a', 'sidewalk', '.']
i=4000 ['a', 'boy', 'in', 'a', 'blue', 'shirt', 'is', 'jumping', 'into', 'a', 'pool', '.']
blue_score - 0=0.5300537412153783
blue_score - 1=0.3199686171272617
blue_score - 2=0.19328410752854727
blue_score - 3=0.11761174524921017
"""