##########################################################################
# Run analogies of the type queen-woman+man=king
# (Paris is to France what ___ is to Germany)
# To test, download the analogy data as per README
##########################################################################

import sys
sys.path.append("/home/user/faiss/")
import faiss
sys.path.append("/home/user/handy_scripts/modules/loadDM/")
import loadDM
import numpy as np

dm_dict=loadDM.readDM(sys.argv[1], True)        #True is for normalisation: faiss calculates cos as inner prod of normalised vectors
d = int(sys.argv[2])        # dimension

def mk_faiss_index():
  vocab=[]
  word_to_index={}
  xb= np.zeros(shape=(len(dm_dict),d)).astype('float32')

  i=0
  for k,v in dm_dict.items():
    vocab.append(k)
    word_to_index[k]=i
    xb[i]=v.astype('float32')
    i+=1

  index = faiss.IndexFlatIP(d)   # build the index
  index.add(xb)                  # add vectors to the index
  return vocab, word_to_index, index


def readAnalogy():
  triples=[]
  gold=[]
  f=open("/home/user/handy_scripts/modules/analogy/word-test.v1.txt",'r')
  for l in f:
    l=l.rstrip('\n')
    items=l.lower().split()
    if len(items) == 4 and all([ i in dm_dict for i in items]):
       triples.append([items[0],items[1],items[3]])
       gold.append(items[2])
  f.close()
  print "Loaded",len(triples),"analogies..."
  return triples, gold


def compute_analogy(triple, gold, index, vocab):
  score = 0
  vec = dm_dict[triple[0]]-dm_dict[triple[1]]+dm_dict[triple[2]]
  xq= np.zeros(shape=(1,d)).astype('float32')
  xq[0]=loadDM.normalise(vec.astype('float32'))
  D, I = index.search(xq, 4)
  for i in range(4):
    #print vocab[I[0][i]], D[0][i]
    #Usual hack to avoid vectors in given triple
    if vocab[I[0][i]] not in triple:
      if vocab[I[0][i]] == gold:
        score=1
      print triple, vocab[I[0][i]], score, "(actual: ", gold, ")"
      break
  return score


vocab, word_to_index, index = mk_faiss_index()
triples, gold=readAnalogy()

accuracy = 0
for i in range(len(triples)):
  accuracy+=compute_analogy(triples[i], gold[i], index, vocab)

print "OVERALL ACCURACY:", float(accuracy) / float(len(triples))
