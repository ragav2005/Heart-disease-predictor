import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Training model
def train_model():
    heart_data = pd.read_csv("heart.csv")
    x = heart_data.drop(columns="target", axis=1)
    y = heart_data["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

# predictor with input data
def predictor(input_data):
    model = train_model()
    np_arr = np.asarray(input_data)
    input_data_reshaped = np_arr.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction
