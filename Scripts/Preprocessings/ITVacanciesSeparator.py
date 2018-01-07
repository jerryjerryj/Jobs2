import pandas

ids = [400644, 240724, 228243, 428438, 270999, 209204]
with open('vacancies.csv', encoding='utf-8') as f:
    content = f.readlines()
lines = [x.strip() for x in content]

result = []

for line in lines:
    id = int(line.split('\t')[0])
    if id in ids:
        result.append(line)

result.sort()

resultFile = open('vacanciesIT.csv', 'w',encoding='utf-8')
for item in result:
  resultFile.write("%s\n" % item)