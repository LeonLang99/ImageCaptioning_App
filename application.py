import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests, zipfile, io
import os, shutil
from random import *

def Load_Images():
  r = requests.get("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", stream=True)
  print(r.ok)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall("./Flickr")
  
  
features = pickle.load(open("images1.pkl", "rb"))
model = load_model('model_9.h5')
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))
  
  
def Make_Folder():
  newpath = 'Images/' 
  if not os.path.exists(newpath):
    os.makedirs(newpath)

  source_dir = 'Flickr/Flicker8k_Dataset'
  target_dir = 'Images/'
    
  file_names = os.listdir(source_dir)
    
  for file_name in file_names:
     shutil.move(os.path.join(source_dir, file_name), target_dir)

  folder = 'Flickr'
  for filename in os.listdir(folder):
     file_path = os.path.join(folder, filename)
     try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
             os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
     except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))
        
if not os.path.exists("Images"):
    Load_Images()
    Make_Folder()
    
images = "Images/"
max_length = 33


def Image_Caption(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

st.title('Image Captioning Group 13')

with st.expander("Introduction"):
  st.write("Hello, we are Leon Lang, Jean Louis Fichtner and Loredana Bratu and we created this app as a part of our business informatics course. On this page you will have the opportunity to see how we developed step by step this project, what was our motivation and is our future vision. But before starting, we have a question:  What do you see in the picture bellow?")
  st.image("https://futalis.de/wp-content/uploads/2020/07/contentbild-hund-fital-1.jpg")
 #with st.expander("Possible solution")
  st.write("Most probably you would say “A dog running on a field”, some may say “a dog with white and black spots” and others might say “a dog on a grass and some yellow flowers”. Definitely all of these captions are correct for this image and there may be some more as well. But the point we want to make is that it is so easy for us, as human beings, to just have a look at a picture and describe it in an appropriate language. But, can you write a computer program that takes an image as input and produces a relevant caption as output?")

  
  
with st.expander("Data Understanding"):
  st.subheader("About or data...")
  st.write("Our dataset consists of over 80000 images with at least 5 captions each, which are from the open-source dataset MS COCO and are randomly sorted. For the training we will not use the whole data set, because it has an enormous storage capacity.")
  st.text("")
  st.write("Therefore, in advantage of time and costs we only used ca. 8.000 pictures to train our data.")
  st.text("")
  st.write("We will not be delimiting our dataset in specific domains, as our purpose is not image classification, but image captioning, so it’s in our best interests to vary the image topics, so that we call achieve a high accuracy.")
  
with st.expander("Here you can try our Image Captoning Program"): 
  st.write("Try it with a random image by pressing the following Button.")
  if st.button('random picture'):
       z = randint(1, 1000)
       pic = list(features.keys())[z]
       image = features[pic].reshape((1,2048))
       x = plt.imread(images+pic)
       st.image(x)
       st.write("Caption:", Image_Caption(image))

