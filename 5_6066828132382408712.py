# import pandas as pd
from nltk.classify import SklearnClassifier
import nltk
import numpy as np
import re
import string
import json
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

def tokenisasi():
	data = pd.read_csv("dataset.csv", quotechar='"', skipinitialspace=True)
	hasil = data.values
	hasil2 = np.array([[i[0],re.sub(r'[^\x00-\x7F]+',"", str(i[1]))] for i in hasil])
	y =np.array(["".join(i for i in j[1] if i not in string.punctuation) for j in hasil2])
	x =np.array(["".join(i for i in j[0] if i not in string.punctuation).split(" ") for j in hasil2])
	np.savetxt("tokensinopsis.txt", y, fmt="%s")

def getstopword(data, stopword):
	result =[]
	for sentence in data:
		words = sentence.lower().split()
		for word in words:
			if word.strip().lower() in stopword:
				words.remove(word)
		result.append(" ".join(words))
	return result

def getting_label(labelfile):
	return np.array(map(lambda x: map(lambda y: y.strip(),x.lower().replace("\"","").split(",")), labelfile))

def frequency(data,labels):
	freq = {}
	words = {}
	for i in range(data.shape[0]):
		for label in labels[i]:
			freq[label] = freq.get(label, {})
			for kata in data[i]:
				words[kata] =  words.get(kata, 0)
				freq[label][kata] = freq[label].get(kata, 0) + 1
	return freq,words

def feature_extract(words, katas):
	features = []
	for data in words:
		feature = []
		j = 0
		for kata in katas:
			feature.append(data.count(kata))
			j += 1
		features.append(feature)
	return features

def load_model(filename):
	classifier_f = open(filename, "rb")
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	return classifier


def aplikasi(modelfilename):
	# words = np.array(map(lambda x: x.lower().split(),np.genfromtxt("input.txt", dtype='string', delimiter="\n")))
	words = [np.genfromtxt("input.txt", dtype='string', delimiter="\n").tostring().lower().split()]
	kata = np.genfromtxt("feature_list.txt", dtype='string', delimiter='\n')
	genre = np.genfromtxt("target_list.txt", dtype='string', delimiter='\n')
	features = feature_extract(words, kata)
	model = load_model(modelfilename)
	hasil = model.predict(features)
	for i in hasil:
		arrresult = np.where(i==1)
		resultstring = ""
		for h in arrresult[0]:
			resultstring += genre[h]+", "
		print resultstring[:-2],"\n"

# cleansing
# stopword = np.genfromtxt("stopword.txt", dtype=None, delimiter="\n")
# data= np.genfromtxt("cleanwords3.txt", dtype='string', delimiter="\n")
# datastopword =getstopword(data, stopword)
# np.savetxt("cleanwords3.txt", datastopword, fmt="%s")

# count frequency
# labeltest = np.genfromtxt('labelfornltk.txt',dtype='int')
# genre, kata = frequency(words,labels)

# cara extract feature
# features = feature_extract(words, katas)

# cara create model
# ovr = OneVsRestClassifier(MultinomialNB())
# ovr.fit(features,labeltest)

# cara save model
# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(ovr, save_classifier)
# save_classifier.close()

# cara load model
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

# cara predict
# hasil = classifier.predict(features)
# for i in hasil:
# 	arrnya = np.where(i==1)
# 	for h in arrnya[0]:
# 		print genre.keys()[h]

# Aplikasi jadi
aplikasi("naivebayes.pickle")