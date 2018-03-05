import nltk, glob, pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def TokenizeSentences(rawSentences, needStemming):
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

            if (token >= 'а' and token <= 'я') and needStemming:
                cleaned_set.append(st.stem(tokenized))
            elif ((token >= 'а' and token <= 'я') or (token >= 'a' and token <= 'z')):
                cleaned_set.append(tokenized)

        if cleaned_set.__len__()>0:
            sentences.append(cleaned_set)
    return sentences


if __name__ == "__main__":

    '''print('БЫЛА ЛИ ПРОВЕДЕНА ОЧИСТКА ДАННЫХ?'
          '\n1. ссылки типо http://www.rbc.ru/magazine/2016/04/56ead0549a79474e4031fc94')'''
    TARGET = '\Vacancies'
    STEMMING  = True
    OUT_EXTENSION = '.p'
    if not STEMMING:
        OUT_EXTENSION = '.ns.p'

    pathSource = 'F:\My_Pro\Python\Jobs2\Data'
    pathTokenized = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized'
    filesPaths = glob.glob(pathSource+ TARGET+"\*.txt")

    totalSentences = []
    for filePath in filesPaths:
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            content = [x.replace('\t',' ') for x in content]
            totalSentences.extend(content)

    tokenized = TokenizeSentences(totalSentences,STEMMING)
    pickle.dump( tokenized, open(pathTokenized+TARGET+OUT_EXTENSION, "wb" ) )

