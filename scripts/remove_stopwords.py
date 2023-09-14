import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 

def remove_stopwords_from_corpus(corpus, stopwords):
    stop_words = set(stopwords.words('english'))
    file1 = open(file=corpus)
    line = file1.read()
    words = line.split()
    for r in words:
        if not r in stop_words:
            appendFile = open('filteredtext.txt','a')
            appendFile.write(" "+r)
            appendFile.close()