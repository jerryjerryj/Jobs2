import nltk, glob, pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def TokenizeSentences(rawSentences):
    sentences = []
    st = nltk.stem.SnowballStemmer('russian')
    for c in rawSentences:
        tokenized_sents = word_tokenize(c)
        cleaned_set = []
        for tokenized in tokenized_sents:
            if tokenized == "":
                break
            tokenized = tokenized.lower()
            token = tokenized[0]
            if tokenized in stopwords.words('russian'):
                continue
            if (token >= 'а' and token <= 'я'):
                cleaned_set.append(st.stem(tokenized))
            elif(token >= 'a' and token <= 'z'):
                cleaned_set.append(tokenized)

        if cleaned_set.__len__()>0:
            sentences.append(cleaned_set)
    return sentences

TARGET = '\Marked'

pathSource = 'F:\My_Pro\Python\Jobs2\Data'
pathTokenized = 'F:\My_Pro\Python\Jobs2\Data\TokenizedSentences'
filesPaths = glob.glob(pathSource+ TARGET+"\*.txt")

totalSentences = []
for filePath in filesPaths:
    with open(filePath, encoding='utf-8') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = [x.replace('\t',' ') for x in content]
        totalSentences.extend(content)

tokenized = TokenizeSentences(totalSentences)
pickle.dump( tokenized, open(pathTokenized+TARGET+'.p', "wb" ) )

