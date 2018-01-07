import gensim, logging
import pandas

def SentenceToVector(wv, sentence):
    vectors = pandas.DataFrame()
    index = 0
    try:
        for word in sentence:
            if word in wv.vocab:
                vectors[index] = wv[word]
            index += 1
        vectors = vectors.transpose()
        vector = vectors.mean().values.tolist()
    except Exception:
        return []
    return vector