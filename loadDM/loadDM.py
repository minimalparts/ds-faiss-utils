#Load semantic space in DM format.

import numpy as np

def normalise(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def readDM(infile, norm):
  dm_dict={}
  f=open(infile,'r')
  for l in f:
    l=l.replace('\t',' ')			#In case tab-separated
    items=l.rstrip('\n').split()
    row=items[0]
    vec=[float(i) for i in items[1:]]
    if norm:
      vec=normalise(vec)
    dm_dict[row]=np.array(vec)
  return dm_dict
