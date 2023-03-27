# import numpy as np
from nltk.corpus import wordnet as wn

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
import sys

try:

    sp = ['football', 'ronaldo', 'cristiano ronaldo', 'uefa', 'score', 'united', 'golden', 'shoes', 'league',
             'transfer', 'final', 'fifa', 'cup', 'league cup', 'world cup', 'goals', 'goal', 'Messi', 'la liga']
    p = ['iran', 'usa', 'terrorism', 'syria', 'national', 'war', 'trump', 'president', 'staff', 'weapon', 'trade',
           'barak obama', 'sanctions', 'boundry', 'france', 'united nations']
    h = ['eat', 'cancer', 'epidemic', 'milk', 'tips', 'apple', 'meal', 'vegetables', 'snack', 'fat', 'body',
              'meat', 'health', 'family', 'oil', 'fitness']
    c = ['hardware', 'software', 'data', 'artificial', 'intelligence', 'hash', 'algorithm', 'cpu', 'hard',
                'big data', 'computer', 'pc', 'mouse', 'internet', 'iot', 'keyboard', 'python', 'c++']
    sc = ['school', 'math', 'student', 'teacher', 'course', 'study', 'homework', 'physics', 'sport']
    concepts = [sp, p, h, c, sc]
    doc_number=10
    concepts_num=5
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    s = []
    cols=range(1, doc_number, 1)
    term_doc = pd.DataFrame(0, index=s, columns=cols)
    index = ['sport', 'plotics', 'health', 'computer', 'school']
    concept_doc= pd.DataFrame(0,index=index, columns=cols)
    for i in range(0, doc_number):
        wordsFiltered = []

        j = i + 1
        # reading dataset
        tx = "dt/" + str(j) + ".txt"
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
                    term_doc.loc[w,i]=1

        print(i,':',wordsFiltered)

        for id in range(0,concepts_num):
            score=0
            print(id)
            for spx in range(0,len(sp)):
                w1=sp[spx]+'.n.01'
                w1x = wn.synset(w1)
                print('hi')
                if w1x:
                    print('by')
                    for wx in range(0,len(wordsFiltered)):
                       print('hi2')
                       w2=wordsFiltered[wx]+'.n.01'
                       w2x=wn.synset(w2)
                       print('hi2')
                       if w2x:
                         print('hi3')
                         sim = w1.wup_similarity(w2)
                         score= sim+score
            print(score)












except :
    print('find error')


