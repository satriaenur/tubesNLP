# import pandas as pd
from nltk.classify import SklearnClassifier
import nltk
import numpy as np
import re
import string
import json
import pickle
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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
		
# Ambil data training untuk diubah jadi data testing
def toDataTest():
	openCSV = open('dataset (3).csv','r' )
	csvReader = csv.reader(openCSV, delimiter=",")
	openCSV	= open('datatest/genreDatetest.csv','wb')
	csvWriter=csv.writer(openCSV,delimiter=',', quotechar='|')
	newList = list(csvReader)
	split_size = int(len(newList) * 0.25)
	newList = newList[1:split_size]
	index = 0
	for row in newList:
		if len(row[len(row) - 1])!=0:
			openTxt = open('datatest/' + str(index) + '.txt', 'wb')
			openTxt.write(row[len(row) - 1])
			csvWriter.writerow(row[0].split(","))
			index += 1
	print "finish"

# prediksi genre untuk tiap datatest
def predictdatatest():
	openCSV = open('datatest/predictionTest.csv', 'wb')
	csvWriter = csv.writer(openCSV, delimiter=',', quotechar='|')
	for i in range(1121):
		filename='datatest/'+str(i)+'.txt'
		csvWriter.writerow(aplikasi("naivebayes.pickle",filename))
		print aplikasi("naivebayes.pickle",filename)
		if i%200==0:
			print "iterasi : "+ str(i)

# Evaluasi Precision and Recall
def evaluation():
	actual = list(csv.reader(open('datatest/genreDatetest.csv', 'r'), delimiter=','))
	prediction=list(csv.reader(open('datatest/predictionTest.csv', 'r'), delimiter=','))
	genres = np.genfromtxt("target_list.txt", dtype='string', delimiter='\n')
	sumClass=len(genres)

	precision = 0
	recal = 0
	for genre in genres:
		index=0
		tp=0
		fn=0
		fp=0
		for row in prediction:
			newRow=[x.lower() for x in row]
			newActual=[x.lower() for x in actual[index]]

			# print newRow
			# print newActual
			# print "-----------------------------------------"
			if (genre in newRow) and (genre in newActual):
				tp+=1
			elif genre in newRow:
				fp+=1
			elif genre in newActual:
				fn+=1
			index+=1

		predicted=tp+fp
		trueCondition=tp+fn

		if predicted==0:
			predicted=1
		if trueCondition==0:
			trueCondition=1

		print "Genre : "+genre
		# print tp
		# print fp
		# print fn
		# print trueCondition
		precision+=tp/(predicted*1.0)
		recal+=tp/(trueCondition*1.0)
		print "Precision : "+str(tp/(predicted*1.0))
		print "Recall : "+str(tp/(trueCondition*1.0))
		print "-----------------------------------------"


	precision=precision/(sumClass*1.0)
	recal=recal/(sumClass*1.0)

	print "ALL EVALUATION"
	print "precision : "+str(precision)
	print "recall : "+str(recal)

# cleansing
# stopword = np.genfromtxt("stopword.txt", dtype=None, delimiter="\n")
# data = np.genfromtxt("cleanwords3.txt", dtype='string', delimiter="\n")
# datastopword =getstopword(data, stopword)
# np.savetxt("cleanwords3.txt", datastopword, fmt="%s")

# cleansing with stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
data = np.genfromtxt("cleanwords3.txt", dtype='string', delimiter="\n")
data = map(lambda x: stemmer.stem(x), data)
print data[1]
np.savetxt("stemmeddata.txt", data, fmt='%s')

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
# aplikasi("naivebayes.pickle")