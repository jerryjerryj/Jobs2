from gensim.models import Word2Vec

path = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsBooks\\'
name = 'Администратор БД.w2v'
model = Word2Vec.load(path+name)

sim1 = 'Мониторинг состояния удаленных баз данных'
sim2 = 'Мониторинг работы БД, сбор статистической информации о работе БД'
notsim = 'Установка обновлений и новых версий'

print(model.wmdistance(sim1,sim2))
print(model.wmdistance(sim1,notsim))
print(model.wmdistance(sim2,notsim))