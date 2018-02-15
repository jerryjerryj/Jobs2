import glob

def GetClasses(pathToMarked):
    filesPaths = glob.glob(pathToMarked + "\*.txt")
    classes = []
    for filePath in filesPaths:
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for c in content:
                classes.append(int(c.split('\t')[0]))
    return classes

def GetMulticlasses(pathToMarked):
    filesPaths = glob.glob(pathToMarked + "\*.txt")
    classes = []
    for filePath in filesPaths:
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for c in content:
                strClasses = c.split('\t')[0].split(';')
                classes.append([int(x) for x in strClasses ])
    return classes

def GetFromPS(markedFilePath):
    result = []

    with open(markedFilePath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    currentClassId = -1

    for line in lines:
        if line[0] !='\t':
            currentClassId+=1
        result.append([currentClassId])
    return result

def GetMulticlassesDemands(markedTestFilePath):
    classes = []
    with open(markedTestFilePath, encoding='utf-8') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for c in content:
            strClasses = c.split('\t')[0].split(';')
            classes.append([int(x) for x in strClasses])
    return classes

#vacanciesMarkedDir = 'F:\My_Pro\Python\Jobs2\Data\Vacancies'
#print(GetMulticlasses(vacanciesMarkedDir)[400:500])