! pip install tensorflow
! pip install h5py
! pip install keras
! pip install nltk
import nltk
nltk.download("all")
import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pandas as pd
import os

files_list = os.listdir('/Users/shiqinchoo2019/Desktop/ndr_model/ndr/')
path = '/Users/shiqinchoo2019/Desktop/ndr_model/ndr/'

files_names_list = []

for i in range(len(files_list)):
    name = path + files_list[i]
    files_names_list.append(name)

len(files_names_list)


texts = []

for i in range(len(files_list)):
    f = open(files_names_list[i], "r", encoding = 'ISO-8859-1')
    text = f.read()
    texts.append(text)

len(texts)

years = []
for i in range(len(files_names_list)):
    year = files_names_list[i][44:48]
    years.append(year)

years

ndr_text = pd.DataFrame()
ndr_text['Year'] = years
ndr_text['Speech'] = texts
ndr_text

all_speech = []
for i in range(len(ndr_text)):
  speech = ndr_text.iloc[i]['Speech']
  all_speech.append(speech)

speech_text = " ".join(all_speech)
len (speech_text)

speechtext = open("speech_text.txt", "w")
speechtext.write(speech_text)
speechtext.close()

import string
 
# turn a doc into clean tokens
def clean_doc(speech_text):
	# replace '--' with a space ' '
	speech_text = speech_text.replace('--', ' ')
	# split into tokens by white space
	tokens = speech_text.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

tokens = clean_doc(speech_text)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
# save sequences to file
out_filename = 'ndr_speech.txt'
save_doc(sequences, out_filename)

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

in_filename = 'ndr_speech.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/tmp/ndr.hdf5', verbose=1)
# fit model
model.fit(X, y, batch_size=128, epochs=500, callbacks=[checkpointer])


# save the model to file
model.save('ndr.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer_ndr.pkl', 'wb'))

from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

# load cleaned text sequences
in_filename = 'ndr_speech.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('ndr.h5')

# load the tokenizer
tokenizer = load(open('tokenizer_ndr.pkl', 'rb'))

# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')

# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 100)
print(generated)


# load cleaned text sequences
in_filename = 'ndr_speech.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('ndr.h5')

# load the tokenizer
tokenizer = load(open('tokenizer_ndr.pkl', 'rb'))

# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')

# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 100)
print(generated)


seed_text = "support for elderly"

generated = generate_seq(model, tokenizer, seq_length, seed_text, 100)
print(generated)





















