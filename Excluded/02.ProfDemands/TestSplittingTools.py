from Scripts.Tools.IO import IOTools
from sklearn.model_selection import train_test_split

def GetCommonSplit():
    iotools = IOTools()
    dataset = iotools.LoadPickle('F:\My_Pro\Python\Jobs2\Scripts\Clustering\ProfDemands\ProfDemands&Classes.p')
    demands = dataset['demands'].tolist()
    classes = dataset['classes']
    df = iotools.LoadPickle(
        'F:\My_Pro\Python\Jobs2\Scripts\Classification\VacanciesTextsToProfessions\DatasetFromMarkedDemands.p')
    test_demands = df['vectors'].tolist()
    test_classes = df['classes'].tolist()
    return demands, test_demands, classes, test_classes

def GetTrainTestSplit(testSize):
    iotools = IOTools()
    dataset = iotools.LoadPickle('F:\My_Pro\Python\Jobs2\Scripts\Clustering\ProfDemands\ProfDemands&Classes.p')
    demands = dataset['demands'].tolist()
    classes = dataset['classes'].tolist()
    df = iotools.LoadPickle(
        'F:\My_Pro\Python\Jobs2\Scripts\Classification\VacanciesTextsToProfessions\DatasetFromMarkedDemands.p')
    demands.extend(df['vectors'].tolist())
    classes.extend(df['classes'].tolist())
    return train_test_split(demands, classes, test_size=testSize)