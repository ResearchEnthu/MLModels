import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open("classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Student Performance")

st.write(f"Predicted Iris Species: {model}")