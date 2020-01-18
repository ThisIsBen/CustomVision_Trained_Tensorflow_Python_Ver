# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:46:14 2020

@author: KNK09087
"""


import tensorflow as tf
import os
from preporecssInputImage import *

def ElMDetectClassify(imageFilePath):
    graph_def = tf.compat.v1.GraphDef()
    labels = []
    
    # These are set to the default names from exported models, update as needed.
    filename = "model.pb"
    labels_filename = "labels.txt"
    
    #Enter the file path of the image to be predicted
    imageFile = imageFilePath
    
    # Import the TF graph
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())
            
    
    # Load the image to be predicted 
    image = Image.open(imageFile)
    
    # Update orientation based on EXIF tags, if the file has orientation info.
    image = update_orientation(image)
    
    # Convert to OpenCV format
    image = convert_to_opencv(image)       
    
    #Handle images with a dimension >1600
    # If the image has either w or h greater than 1600 we resize it down respecting
    # aspect ratio such that the largest dimension is 1600
    image = resize_down_to_1600_max_dim(image)
    
    
    #Crop the largest center square
    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)
    
    #Resize down to 256x256
    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)
    
    
    
    #Crop the center for the specific input size for the model
    # Get the input size of the model
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]
    
    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
    
    
    
    #Predict an image
    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'
    
    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
        except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)
            
    #View the results
        # Print the highest probability label
        highest_probability_index = np.argmax(predictions)
        print('Classified as: ' + labels[highest_probability_index])
        print()
    
        # Or you can print out all of the results mapping labels to probabilities.
        label_index = 0
        for p in predictions:
            truncated_probablity = np.float64(np.round(p,8))
            print (labels[label_index], truncated_probablity)
            label_index += 1
    
    
    
