It explains the scripts present in the source-code folder

"ALIGN_DATASET_MTCNN.PY"


* This script imports two files detect_face.py and facenet.py
* detect_face.py script performs MTCNN operation for detecting faces.
* facenet.py script performs creation of directory based on labels.
* it creates aligned face directory by using MTCNN.
CONSTRAINS:
* if there is no face detected, it skip the alignment process.
        or
*if the image is having the dimension less than two, it will skip the alignment process.

input <- unaligned images
output -> aligned images(182*182)


"CLASSIFIER.PY"

* this script is used for traing the classifier.
* load the dataset[load_data]using facenet script.
* it loads the pretrained model.
* extract  the features of the images using the pretrained model.
* saves the features as embedding.
* embedding are used for training the svm classifier.
* saves the trained classifier in pickle format.

input <- aligned images
output -> classifier




"PREDICT.PY"

* this script loads and align the images. It resizes to the format in which, we can able to feed to neural network for feature extraction purpose.
* pretrained model is loaded for feature extraction.
* trained model  [classifier] is loaded for prediction purpose.

input <- aligned_images
output -> prediction result




"FAST-PRED.PY"

* this script import mtcnn_detector script and helper script for detecting faces from the video file.
* pretrained model is loaded.
* resizes the detected face to 182*182 and then resized to 160*160.
* the resized images are passed to prewhiten using facenet file for feature extraction using pretrained model and predicts the label using trained classfier

input <- video_file
output -> predictions



