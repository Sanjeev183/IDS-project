import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    le = LabelEncoder()
    data['protocol_type'] = le.fit_transform(data['protocol_type'])
    data['service'] = le.fit_transform(data['service'])
    data['flag'] = le.fit_transform(data['flag'])
    data['label'] = le.fit_transform(data['label'])

    X = data.drop('label', axis=1)
    y = data['label']

    return X, y
