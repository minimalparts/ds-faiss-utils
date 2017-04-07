#Evaluate semantic space against MEN dataset
#USAGE: python ./MEN.py [space]
#EXAMPLE: python MEN.py ~/Corpora/composes-vectors/EN-wform.w.5.cbow.neg10.400.subsmpl.txt

import sys
sys.path.append("/home/user/handy_scripts/modules/loadDM/")
import loadDM
from scipy import stats
import numpy as np
from math import sqrt

#Note: this is scipy's spearman, without tie adjustment
def spearman(x,y):
	return stats.spearmanr(x, y)[0]

def readMEN():
  pairs=[]
  humans=[]
  f=open("/home/user/handy_scripts/modules/MEN/MEN_dataset_natural_form_full",'r')
  for l in f:
    l=l.rstrip('\n')
    items=l.split()
    pairs.append((items[0],items[1]))
    humans.append(float(items[2]))
  f.close()
  return pairs, humans

def cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        print len(peer_v),len(query_v)
        raise ValueError("Vectors must be of same length")
    num = np.dot(peer_v, query_v)
    den_a = np.dot(peer_v, peer_v)
    den_b = np.dot(query_v, query_v)
    return num / (sqrt(den_a) * sqrt(den_b))


def compute_spearman(infile):
  dm_dict=loadDM.readDM(infile, True)
  pairs, humans=readMEN()

  system_actual=[]
  human_actual=[]			#This is needed because we may not be able to calculate cosine for all pairs
  count=0

  for i in range(len(pairs)):
    human=humans[i]
    a,b=pairs[i]
    if a in dm_dict and b in dm_dict:
      cos=cosine_similarity(dm_dict[a],dm_dict[b])
      system_actual.append(cos)
      human_actual.append(human)
      count+=1


  sp = spearman(human_actual,system_actual)	
  print "SPEARMAN:",sp, "(calculated over",count,"items.)"

compute_spearman(sys.argv[1])
