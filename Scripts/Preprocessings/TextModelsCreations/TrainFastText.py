from gensim.models.fasttext import FastText
from Scripts.Preprocessings.TextModelsCreations.Tools import  CollectSentences, PrintSourceDataStats

pathSource = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized\\NS'
pathOut = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models'

sentences = CollectSentences(pathSource)

model = FastText(min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

model.save(pathOut+'\\FastText.model')