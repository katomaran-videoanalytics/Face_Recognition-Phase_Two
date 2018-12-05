# AUTHOR - SAI SUBRAMANI

import tensorflow as tf
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import facenet1
import facenet
import os
import sys
import math
import cv2
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from six.moves import xrange
import mxnet as mx
from mtcnn_detector import MtcnnDetector

#"rtsp://admin:admin0864@103.60.63.138:8081/cam/realmonitor?channel=1&subtype=1"

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
camera = FileVideoStream("id1.mp4").start()
fps = FPS().start()


font = cv2.FONT_HERSHEY_SIMPLEX


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



with tf.Graph().as_default():
    with tf.Session() as sess:
    	#path for modeldir

        facenet1.load_model("modeldir")

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
        sampleNum = 0
        j = 0
        while True:
            frame = camera.read()
            frame = imutils.resize(frame,width=640)
            if j % 3 == 0:
               # cv2.imwrite("Frame/User."+str(id)+"."+str(sampleNum)+".jpg",frame) 
            

                if frame is None:
                    raise SystemError('issue in grabbing frame')

   #crop th faces
                results = detector.detect_face(frame)
                if results is None:
                    continue

                total_boxes = results[0]
                points = results[1]
                img_list=[]
                for b in total_boxes:
                    x1 = int(b[0])
                    y1 = int(b[1])
                    x2 = int(b[2])
                    y2 = int(b[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
                
                    img = frame[y1:y2,x1:x2]
                    
                    sampleNum = sampleNum+1
                    img = misc.imresize(img, (182, 182), interp='bilinear')
                    aligned = misc.imresize(img, (160, 160), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)        
                    images = np.stack(img_list)
 
            
                    feed_dict = { images_placeholder: images , phase_train_placeholder:False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    classifier_filename_exp = os.path.expanduser("phase2.pkl")
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                    print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                    predictions = model.predict_proba(emb)
                    best_class_indices = np.argmax(predictions, axis=1) 
                    x23=str(class_names[best_class_indices[0]])
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    prob = float(best_class_probabilities[0])
                    x24 = "{:.2f}".format(prob)
                    if float(x24) >= float(0.75):
                        print("name :",x23)
                        print("prob :",x24)
                        #cv2.putText(frame, x23, ((x1,y1-10)), font,1,(0,0,255),2)
                        #cv2.putText(frame, x24, (x2,y2+10), font,1,(0,0,255),2)
                        cv2.imwrite("predictioni/"+x23+str(sampleNum)+".jpg",aligned)
                        sampleNum += 1

            cv2.imshow("detection result",frame )
                    
            j += 1
            k = cv2.waitKey(1)
            if k == ord('x'):
                break

        fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        cv2.destroyAllWindows()
        camera.stop()