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

features = pickle.load(open("images1.pkl", "rb"))
model = load_model('model_9.h5')

r = requests.get("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", stream=True)
print(r.ok)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./Flickr")

newpath = 'Images/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    source_dir = '/Flickr/Flicker8k_Dataset'
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


images = "Images/"
max_length = 33
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))

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

with st.expander("The Team"):
  st.write("Hello, we are Leon Lang, Jean Louis Fichtner and Loredana Bratu and we create this app as a part of our business informatics course ")
with st.expander("The Mission"):
  st.write("The purpose of our app is to automatically describe an image with one or more natural language sentences. To generate textual descriptions of images we will use Machine Learning and Deep Learning Techniques.")
with st.expander("The Dataset"):
  st.write("Here you can see some examples from our Dataset")
  col1, col2, col3 = st.columns(3)

with col1:
    st.header("Chrysler Logo")
    st.image("https://img1.d2cmedia.ca/cb5bf24a74832ba/1471/7214770/C/Chrysler-200-2016.jpg")

with col2:
    st.header("NIKE Shoe")
    st.image("https://img.alicdn.com/imgextra/i3/817462628/O1CN01eLHBGX1VHfUMBA1du_!!817462628.jpg")

with col3:
    st.header("Girl in a white dress")
    st.image("https://static2.yan.vn/YanNews/2167221/202004/co-luc-na-trat-duoc-khen-nuc-no-vi-qua-de-thuong-nho-tang-can-93c37ecb.jpeg")
    
with st.expander("Random Picture"):
  st.write("Please press the following Button to get a random picture from our dataset.")
  if st.button('random button'):
     st.balloons()
  else:
     st.write('not pressed yet')
    
with st.expander("Our vision..."):    
   st.subheader("What are our project objectives?")
   st.write("Our main goal is our app to automatically generate captions, also known as textual descriptions, for random images. The dataset will be in the form [image → captions]. It will consist of input images and their corresponding output captions which have to be as precise as possible, but also short and concise. The caption generator will involve the dual techniques from computer vision - to first understand the content of the image, and a language model from the field of natural language processing to turn the understanding of the image into words in the right order and correct structure.")
   st.subheader("Which problem do we want to solve?")
   st.write("Image captioning can be used in lots of virtual domains such as for virtual assistants, picture recommendations, for image indexing, for social media, but also to explore the world around you, using only your phone's camera which scans a real object and tell someone what kind of object that is (for example Google Lens).")
   st.subheader("About our data…")
   st.write("Our dataset consists of X images which are randomly sorted. For the training we will not use the whole data set, because it has an enormous storage capacity (~ 240TB of data).Therefore, in advantage of time and costs we will use between 500-1000 pictures to train our data. We will delimit our data set to the topic 'public and urban ways of travel', as we think this is a suitable domain to start with when training your data.")
  
with st.expander("Here you can try our Image Captoning Program"):
  st.write("Please upload an image press the following Button.")
  image = st.file_uploader("Choose a file")
  if st.button('Start now'):
    print("Caption:", Image_Caption(image))
  else:
     st.write('not pressed yet')
  st.write("Or try it with an image and press the following Button.")
  if st.button('random picture'):
       z = randint(1, 100)
       pic = list(features.keys())[z]
       image = features[pic].reshape((1,2048))
       x = plt.imread(images+pic)
       plt.imshow(x)
       plt.show()
       print("Caption:", Image_Caption(image))
