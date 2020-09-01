#import all the things
from img_capturer import capture_img
from img_saver import save_img
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import sys
import numpy as np
import pickle
import glob
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
np.set_printoptions(threshold=sys.maxsize)

#***************************************
#*********triplet loss func
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    #Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    #Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    #subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    #Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

#**********preparing model
print("Loading....")
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

#********preparing database
def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        if(identity!="sample"):
            database[identity] = img_to_encoding(file, FRmodel)
    with open('faceEncodingDataset.dat', 'wb') as f:
        pickle.dump(database,f)
#*******who_is_it function
def who_is_it(image_path,model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    Arguments:
    image_path -- path to an image
    model -- your Inception model instance in Keras
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    with open('faceEncodingDataset.dat','rb') as f:
        database=pickle.load(f);
    ## Compute the target "encoding" for the image.
    encoding = img_to_encoding(image_path, model)
    ## Find the closest encoding ##
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)
        print(name,dist)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
            
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity
#********the interface*******

while True:
    print("\n**************MENU*************")
    print("1.prepare database")
    print("2.Enroll a new person into the database")
    print("3.Authenticate a person")
    print("4.Quit")
    choice=int(input())
    if choice==1:
        prepare_database()
        print("Database prepared.")
    elif choice==2:
        K.set_image_data_format("channels_last")
        save_img()
        K.set_image_data_format("channels_first")
    elif choice==3:
        K.set_image_data_format("channels_last")
        capture_img()
        K.set_image_data_format("channels_first")
        who_is_it("images/sample.jpg",FRmodel)
    else:
        break
