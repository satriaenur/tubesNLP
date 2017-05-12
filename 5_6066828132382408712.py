# import pandas as pd
import numpy as np
import re
import string
import json

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
	for i in range(data.shape[0]):
		for label in labels[i]:
			freq[label] = freq.get(label, {})
			for kata in data[i]:
				freq[label][kata] = freq[label].get(kata, 0) + 1
	return freq

# cleansing
# stopword = np.genfromtxt("stopword.txt", dtype=None, delimiter="\n")
# data= np.genfromtxt("cleanwords3.txt", dtype='string', delimiter="\n")
# datastopword =getstopword(data, stopword)
# np.savetxt("cleanwords3.txt", datastopword, fmt="%s")

# count frequency
words = np.array(map(lambda x: x.split(),np.genfromtxt("cleanwords3.txt", dtype='string', delimiter="\n")))
label = getting_label(np.genfromtxt("Genre.csv",dtype="string",delimiter="\n"))
freq = frequency(words,label)

# for (label, word) in freq.items():
# 	with open('datafreq/'+label+'.json', 'w') as f:
# 		json.dump(word,f)
# with open('frequency.json', 'w') as f:
#     json.dump(freq, f)