#Calculate nearest neighbours of all items in a space.
#USAGE: python neighbours.py [dm file] [word file] [dimensionality] [number neighbours]
#EXAMPLE: python neighbours.py /home/user/Corpora/composes-vectors/EN-wform.w.5.cbow.neg10.400.subsmpl.txt words.txt 400 10

import sys
sys.path.append("/home/user/handy_scripts/modules/loadDM/")
sys.path.append("/home/user/faiss/")
import loadDM
import faiss
import time
import numpy as np

d = int(sys.argv[3])        # dimension
num_n = int(sys.argv[4])    # number of nearest neighbours

#Read space file and do some initialisation
dm_dict=loadDM.readDM(sys.argv[1], True)	#True is for normalisation: faiss calculates cos as inner prod of normalised vectors
nb = len(dm_dict)          		        # database size

vocab=[]
word_to_index={}
xb= np.zeros(shape=(nb,d)).astype('float32')

#Build the index
i=0
for k,v in dm_dict.items():
  vocab.append(k)
  word_to_index[k]=i
  xb[i]=v.astype('float32')
  i+=1

index = faiss.IndexFlatIP(d)   # build the index
index.add(xb)                  # add vectors to the index
#print index.ntotal

#Read word file (those we want nearest neighbours for)
word_file=open(sys.argv[2],'r')
test_words=word_file.read().splitlines()
word_file.close()
print "Found",len(test_words),"words." 

#Put words in matrix xq
i=0
xq= np.zeros(shape=(len(test_words),d)).astype('float32')
for w in test_words:
  if w in word_to_index:
    a=xb[word_to_index[w]]
    xq[i]=loadDM.normalise(a.astype('float32'))
    i+=1

print "Searching index..."
D, I = index.search(xq, num_n)
print "Finished..."

dist_sum=0.0
count=0
for w in range(len(D)):				#Iterate through all query words
  print w
  for i in range(1,len(D[w])):			#Iterate through neighbours for a particular word
    print vocab[I[w][0]], vocab[I[w][i]], D[w][i]
    dist_sum+=D[w][i]
    count+=1

