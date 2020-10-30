__author__ = 'xead'
import joblib
from sklearn.linear_model import LogisticRegressionCV

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("pkl/Logistic_Model.pkl")
        self.vectorizer = joblib.load("pkl/n_gramVectorizer.pkl")
        self.classes_dict = {0: "negative", 1: "positive", -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            print(vectorized)

        except:
            print("prediction error")
            return -1, 0.8
        predict =  self.model.predict(vectorized)[0]
        proba = self.model.predict_proba(vectorized).max()
        print('predict - {0}; proba - {1}'.format(predict, proba))
        print(type(predict),type(proba))
        return predict, proba


    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print('prediction error')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0].copy()
        prediction_probability = prediction[1].copy()
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]