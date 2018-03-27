import nltk, glob, pickle, time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem
from Scripts.Preprocessings.Tokenize import TokenizeSentences

def TokenizeSentencesLemmatized(rawSentences, needStemming):
    sentences = []
    #st = nltk.stem.SnowballStemmer('russian')
    m = Mystem()
    for c in rawSentences:

        tokenized_sents = m.lemmatize(c)
        cleaned_set = []
        for tokenized in tokenized_sents:
            if tokenized == "":
                break
            tokenized = tokenized.lower()
            if tokenized in stopwords.words('russian'):
                continue

            token = tokenized[0]
            if (token >= 'а' and token <= 'я') and needStemming:
                cleaned_set.append(tokenized)
            elif ((token >= 'а' and token <= 'я') or (token >= 'a' and token <= 'z')):
                cleaned_set.append(tokenized)

        if cleaned_set.__len__()>0:
            sentences.append(cleaned_set)
    return sentences


def TokenizeSentencesLemmatizedFaster(rawSentences):
    sentences = []
    m = Mystem()
    lemmatized = m.lemmatize('\n'.join(rawSentences))
    sentence = []
    for token in lemmatized:
        if token is not '\n':
            if token == "":
                return None
            token = token.lower()
            if token in stopwords.words('russian'):
                return None
            letter = token[0]
            if ((letter >= 'а' and letter <= 'я') or (letter >= 'a' and letter <= 'z')):
                return token
            if token is not None:
                sentence.append(token)
        elif sentence is not []:
            sentences.append(sentence)
            sentence = []
    return sentences

filePath = 'F:\My_Pro\Python\Jobs2\Data\Wiki\Администратор БД (additional)'
totalSentences = []
with open(filePath, encoding='utf-8') as f:
    content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.replace('\t', ' ') for x in content]
    totalSentences= content
totalSentences = totalSentences
print(totalSentences)


start = time.time()
tokenized1 = TokenizeSentencesLemmatized(totalSentences,True)
end = time.time()
print('Old variant time: '+str(end-start))

start = time.time()
tokenized2 = TokenizeSentencesLemmatizedFaster(totalSentences)
end = time.time()
print('Old variant time: '+str(end-start))
print(tokenized1)
print(tokenized2)