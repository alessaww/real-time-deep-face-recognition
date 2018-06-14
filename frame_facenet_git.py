from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

from jsonsocket import Client, Server
import sys

import pdb

error_data = '*'
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'det_facenet')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = ['kaori', 'matei', 'nicola', 'sohaib']
        HumanNames = sorted(HumanNames)

        print('Loading feature extraction model')
        modeldir = 'models/20170512-110547/20170512-110547.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = 'models/four20180614.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        print('Start Recognition!')

        host = 'localhost'
        port = 10003

        server = Server(host, port)
        while True:
            data = server.accept().recv()
            print('ReID receives "%s"' % data)

            find_results = []

            frame = cv2.imread(data['path'])

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]

            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            # print('Detected_FaceNum: %d' % nrof_faces)

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is inner of range!')
                        server.send(error_data, sendSize=False)
                        continue
                    
                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    z = len(cropped) - 1
                    cropped[z] = facenet.flip(cropped[z], False)
                    
                    scaled.append(misc.imresize(cropped[z], (image_size, image_size), interp='bilinear'))
                    scaled[z] = cv2.resize(scaled[z], (input_image_size, input_image_size),
                                            interpolation=cv2.INTER_CUBIC)
                    scaled[z] = facenet.prewhiten(scaled[z])
                    scaled_reshape.append(scaled[z].reshape(-1, input_image_size, input_image_size, 3))
                    feed_dict = {images_placeholder: scaled_reshape[z], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    # boxing face

                    # plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    print('Result: ', best_class_indices[0], ' Name: ', HumanNames[best_class_indices[0]], 'best class probab', int(best_class_probabilities*100)) # not useful in this case (only 2 people) cause the other one is 100 - best_class, ' scores: ', predictions )
                    best_probab = int(best_class_probabilities*100)
                    result_names = HumanNames[best_class_indices[0]] + ' ' + str(best_probab)
                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), thickness=1, lineType=2)

                    text_fps_y = 20
                    text_fps_x = 20
                    strr = 'test'

                    cv2.putText(frame, strr, (text_fps_x, text_fps_y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                    # c+=1
                    cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        server.send(error_data, sendSize=False)
                        break

                    # #video writer
                    # out.release()
                    cv2.destroyAllWindows()

                send_data = HumanNames[best_class_indices[0]]
                print('ReID sends "%s"' % send_data)
                server.send(send_data, sendSize=False)        


            else:
                print('Unable to align')
                server.send(error_data, sendSize=False)