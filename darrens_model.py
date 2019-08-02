import sys, csv
import pandas as pd

def darrens_model():
    df_bodies = pd.read_csv("data/train_bodies.csv")
    df_stances = pd.read_csv("data/train_stances.csv")

