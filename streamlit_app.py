import tensorflow as tf
import numpy as np
import json
import streamlit as st
import pandas as pd
import time

st.set_page_config("Drug Toxicity Classifier", page_icon="icon.jfif")
st.title("Drug Toxicity Classifier")
model=tf.keras.models.load_model("model_clintox-0.9130_us.h5")
with open("vocab_tox.json", "r") as f:
    voc=json.load(f)
    

def encode(row):
  r2=row
  for i in range(len(row)):
    try:
        r2[i]=voc[row[i]]
    except:
        r2[i]=1
  return r2
def one_hot(x):
  x3d=np.zeros((x.shape[0], x.shape[1], len(voc)+1))
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x3d[i][j][x[i][j]]=1
  
  return x3d
def predict(d):
  l=list(d)
  e=encode(l)
  e=np.expand_dims(e, axis=0)
  p=x_train=tf.keras.preprocessing.sequence.pad_sequences(e, 210)
  oh=one_hot(p)
  pred=model.predict(oh)
  p=np.argmax(pred, axis=-1)
  return p, pred

drug=st.text_input("Enter Drug Smiles")

if st.button("Check"):
    p, pred=predict(drug)
    pr=round(float(pred[0][1])*100, 2)
    sts="Safe" if pr>60 else "Unsafe"
    if sts=="Safe":
        st.balloons()
    st.metric("Safety", sts, delta=round(pr-60, 2), delta_color='normal')
    
table=[
    ["H2S04", "OS(=O)(=O)O"],
    ["Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"],
    ["Paracetamol", "CC(=O)NC1=CC=C(C=C1)O"],
    ["Ozone", "[O-][O+]=O"]
    ]
dat=pd.DataFrame(table, columns=["Drug", "SMILES"])

st.text("Example")
st.table(dat)



