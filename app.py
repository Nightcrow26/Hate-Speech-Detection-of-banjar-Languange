import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def main():
    st.title("TF-IDF dan SVM Klasifikasi Ujaran Kebencian Bahasa Banjar")
    news_text = st.text_area("Enter News Here", "Type Here")
    prediction_labels = {'Ujaran Kebencian': 0, 'Bukan Ujaran Kebencian': 1}
    if st.button("Classify"):
        st.text("Original Text::\n{}".format(news_text))
        vect_text = TfidfVectorizer.transform([news_text])
        with open('model.pkl', 'rb') as f:
             predictor = pickle.load(f)
        prediction = predictor.predict(vect_text)
        final_result = get_key(prediction, prediction_labels)
        st.success("News Categorized as:: {}".format(final_result))

if __name__ == '__main__':
	main()