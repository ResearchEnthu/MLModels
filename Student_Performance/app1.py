import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open("classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)