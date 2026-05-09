import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_lstm.h5", compile=False)

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

index_word = {index: word for word, index in tokenizer.word_index.items()}

def predict_next_word(model, tokenizer, text):
    max_sequence_len = model.input_shape[1] + 1

    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    return index_word.get(predicted_word_index, "Word not found")

st.title("Next Word Prediction with LSTM")

input_text = st.text_input(
    "Enter the sequence of words",
    "to be or not to be"
)

if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text)
    st.write(f"Next word: {next_word}")
