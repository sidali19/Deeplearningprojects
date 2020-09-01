import numpy as np
import _pickle as pickle
import pandas as pd

maxLCap = 35
nbkeep = 100

filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nbTrain = df.shape[0]
iter = df.iterrows()

caps = []  # Set of captions
imgs = []  # Set of images
for i in range(nbTrain):
    x = iter.__next__()
    caps.append(x[1][1])
    imgs.append(x[1][0])

outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
[listwords, embeddings] = pickle.load(open(outfile, "rb"))  # Loading reduced dictionary
indexwords = {}  # Useful for tensor filling
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

# Loading images features
encoded_images = pickle.load(open("encoded_images_PCA.p", "rb"))


# Allocating data and labels tensors
tinput = 202
tVocabulary = len(listwords)
X_train = np.zeros((nbTrain, maxLCap, tinput))
Y_train = np.zeros((nbTrain, maxLCap, tVocabulary), bool)

for i in range(nbTrain):
    words_in_caption = caps[i].split()
    indseq = 0  # current sequence index (to handle mising words in reduced dictionary)
    for j in range(len(words_in_caption) - 1):
        current_w = words_in_caption[j].lower()
        if current_w in listwords:
            X_train[i, indseq, 0:102] = embeddings[listwords.index(current_w)] # COMPLETE WITH YOUR CODE
            X_train[i, indseq, 102:202] = encoded_images[imgs[i]] # COMPLETE WITH YOUR CODE

        next_w = words_in_caption[j + 1].lower()
        if next_w in listwords:
            index_pred = listwords.index(next_w) # COMPLETE WITH YOUR CODE
            Y_train[i, indseq, index_pred] = 1 # COMPLETE WITH YOUR CODE
            indseq += 1  # Increment index if target label present in reduced dictionary

outfile = 'Training_data_' + str(nbkeep)
np.savez(outfile, X_train=X_train, Y_train=Y_train)  # Saving tensor
