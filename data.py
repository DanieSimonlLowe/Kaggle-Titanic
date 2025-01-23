import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler

def loadData(dataset):
    classes = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    base = dataset[classes]
    # has cabin 
    for col in ["Age", "Fare"]:
        base.loc[:, col] = base[col].fillna(base[col].median())

    # Fill missing categorical data with mode
    for col in ["Embarked"]:
        base.loc[:, col] = base[col].fillna(base[col].mode()[0])

    # Add binary indicators for missing values
    base.loc[:, "has_cabin"] = dataset["Cabin"].notna().astype(int)

    return base

scaler = None

def loadDataNum(dataset):
    classes = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    base = dataset[classes]
    # has cabin 
    for col in ["Age", "Fare"]:
        base.loc[:, col] = base[col].fillna(base[col].median())



    # Fill missing categorical data with mode
    for col in ["Embarked"]:
        base.loc[:, col] = base[col].fillna(base[col].mode()[0])

    base.loc[:, "Sex"] = base["Sex"].map({"male": 0, "female": 1})
    base = pd.get_dummies(base, columns=["Embarked"], prefix="Embarked")

    # Add binary indicators for missing values
    base.loc[:, "has_cabin"] = dataset["Cabin"].notna().astype(int)

    global scaler
    if (scaler == None):
        scaler = MinMaxScaler()
        base = pd.DataFrame(scaler.fit_transform(base), columns=base.columns)
    else:
        base = pd.DataFrame(scaler.transform(base), columns=base.columns)

    return base

def getTrain():
    train_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\train.csv')
    y = train_data["Survived"]
    x = pd.get_dummies(loadData(train_data))

    return x, y

def getTest():
    test_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\test.csv')
    X_test = pd.get_dummies(loadData(test_data))
    return X_test

def getTrainNum():
    train_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\train.csv')
    y = train_data["Survived"]
    x = pd.get_dummies(loadDataNum(train_data))

    return x, y

def getTestNum():
    test_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\test.csv')
    X_test = pd.get_dummies(loadDataNum(test_data))
    return X_test