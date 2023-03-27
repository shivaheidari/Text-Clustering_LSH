# get articles as docs
#  make n-gram set
# hash them by bloom filter and make the characteristic matrix item-doc
# bloom filter is used for making characteristic matrix. if an n-gram is in the documnet. choose a suitable size for hash table.
# make signature matrix b=20 and r=5
# lsh produces candidate pairs.
# output: similar documnets should be clusterd to the same cluster.

########################## IMPORTS ###############################
import pandas as pd
import sys
import os.path
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import hashlib
from random import *
import time





########################## DEFINES ###############################

def word2ngrams(text, n, exact=True):
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]
# hash_function 1
hash_array_size=100000
def fnvhash(s):
    hval = 0x811c9dc5 # Magic value for 32-bit fnv1 hash initialisation.
    fnvprime = 0x01000193
    fnvsize = 2**32
    if not isinstance(s, bytes):
        s = s.encode("UTF-8", "ignore")
    for byte in s:
        hval = (hval * fnvprime) % fnvsize
        hval = hval ^ byte
    return hval % hash_array_size
# hash_2
def md(s):
    s = s.encode("UTF-8", "ignore")
    hashed = hashlib.md5(s)
    hashed2 = hashed.hexdigest()
    return (int(hashed2,16)) % hash_array_size

########################## MAIN ###############################

#  preprocessing steps
try:
    start=time.time()
    clusters_num=3
    clusters_num=np.zeros(clusters_num)
    shingle = " "
    pairs = dict()
    fp = 0
    sentence = " "
    doc_number = 4
    cols = range(1,doc_number, 1)
    terrm_shingle=[]
    mh_number = 60
    term_doc=pd.DataFrame(0,index=terrm_shingle,columns=cols)
    US = set()  # universal set
    j = 0
    hash_array=np.zeros(hash_array_size)
    stopWords = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    # ####################read documents and normalizing###############################33
    for i in range(0, doc_number):
        wordsFiltered = []
        n_grams_file=[]
        j = i + 1
        # reading dataset
        tx = "dataset/" + str(j) + ".txt"
        # print(tx)
        f = open(tx)
        raw = f.read().split("\n")
        for line in raw:
            # tokenize
            line = line.lower()
            l = tokenizer.tokenize(line)
            for w in l:
                # stopwords
                if w not in stopWords:
                    # stemming
                    w = ps.stem(w)
                    wordsFiltered.append(w)
        sentence=''.join(wordsFiltered)
        # word filtered is all words in the document in normal form
        # n_grams_file is set of ngrams of the words in document
        # us is set of ngrams of the corpus. o(n2)
        n_grams_file = word2ngrams(sentence,6)
        US = US.union(n_grams_file)
        print("universal n_grams", US)
        print('lenngram',len(n_grams_file))
        print('lenus',len(US))

        # #################bloom filter ##############################
        for index in range(0, len(n_grams_file)):
             shingle = n_grams_file[index]
             hash1 = fnvhash(shingle)
             hash2 = md(shingle)
             if(hash_array[hash1]==0) & (hash_array[hash2]==0):  # shingle is new
                #  add shingle to dataframe(term /doc)
                hash_array[hash1] = 1
                hash_array[hash2] = 1
                term_doc.loc[shingle,j] = 1
             else:
                 # calculate false positive
                 if (shingle in term_doc.index):
                   term_doc=term_doc.set_value(shingle,j,1)
                 else:
                     fp=fp+1
        #        set value of term doc =1

    print('lenght:', len(US), '\n', US)
    print('false positive', fp)


    ngram_num = len(US)
    print('lenght:',len(US),'\n',US)
    # print('number of ones',np.count_nonzero(hash_array))
    # print('number of ones in term_doc',np.count_nonzero(term_doc))
    print('false positive',fp)
    ngram_num= len(US)
    ngram_num_list=list(range(0,ngram_num))

    # ###################hash matrix ################################3
    hash_matrix = np.zeros((mh_number,len(ngram_num_list)),dtype=int)
    # print(hash_matrix.shape)
    for tx in range(1,mh_number):
        shuffle(ngram_num_list)
        for jx in range(0,len(ngram_num_list)):
            hash_matrix[tx][jx]=ngram_num_list[jx]
    # ###############hash matrix has been built#################3






    total_rows=len(term_doc.axes[0])
    total_cols=len(term_doc.axes[1])
    term_doc_arr= term_doc.values
    sig=np.full([mh_number,total_cols],np.inf)



    #################building signature matrix#############
    for ind in range(0, total_rows):
        for h in range(0, mh_number):
            hi = hash_matrix[:, ind]
            for cx in range(0, total_cols):
                if term_doc_arr[ind][cx] == 1:
                    for h2 in range(0, len(hi)):
                        if sig[h2][cx] > hi[h2]:
                            sig[h2][cx] = hi[h2]

    print("signature matrix:\n", sig)
    print(sig.shape)
#     #############signature martix has beenn built########################


###########pairs######################
    r=2
    b=mh_number/r
    print('number of bands',b)
    candidates = set()
    band= np.zeros((r,total_cols))
    for ind in range(0, mh_number, r):
      pairs = dict()
      band[0:r, 0:total_cols] = sig[ind:ind + 2, :]
      print('band: \n', ind, band)
       # #############band has been made correctly


      for i in range(0,r):
       for j in range(0,total_cols):
         jx=j+1
         while jx<total_cols:
           if band[i][j]==band[i][jx]:
             ik=min(j,jx)
             jk=max(j,jx)
             k=str(ik+1)+str(jk+1)
             if k not in pairs:
                 pairs[k]=0

             pairs[k]+=1
           jx = jx + 1
      print(pairs)


      for k in pairs.keys():
        v = pairs[k]
        # print(k, v)
        if v == 2:
            candidates.add(k)

    print('candidate pairs',candidates)
    print('number of elements', len(US))




except ValueError:

        print(ValueError)
#
# except ValueError:
#     print(ValueError)
#
