import numpy as np
import pickle

class insurance():

    def __init__(self,data):
        self.data = data

    def model_load(self):
        with open(r'artifacts/model.pkl','rb') as file:
            self.model = pickle.load(file)

    def predict(self):
        self.model_load()

        age = float(self.data["age"])
        gender = float(self.data["gender"])
        bmi = float(self.data["bmi"])
        children = float(self.data["children"])
        smoker = int(self.data["smoker"])
        
        array = np.array([age,gender,bmi,children,smoker], ndmin=2)
        prediction = self.model.predict(array)[0]
        prediction
        return prediction

if __name__ == "__main__":
    age = 53.0
    gender = 1.0
    bmi = 26.6
    children = 0
    smoker = 0

    array = np.array([age,gender,bmi,children,smoker], ndmin=2)

    insurance_obj = insurance(array)
    insurance_obj.predict()