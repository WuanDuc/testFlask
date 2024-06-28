from app import app
from flask import make_response, render_template, send_file
from flask import Flask, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
#from tensorflow.python.keras.models import load_model
from keras.models import load_model
#from keras.api._v2.keras.models import load_model
from keras.layers import BatchNormalization
import tensorflow as tf

import urllib.request
from io import BytesIO
import base64
import os
import sys
import cv2
import numpy as np
from base64 import b64decode
import imutils
import shutil
import time
import cloudinary.uploader
cloudinary.config( 
  cloud_name = "dpej7xgsi", 
  api_key = "528711498628591", 
  api_secret = "zE8uzpVsTalZQmpeRHOvUnc81Fw",
)

from cloudinary import uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
angry_count = 0
disgust_count = 0
fear_count = 0
happy_count = 0
neutral_count = 0
sad_count = 0
surprise_count = 0


def draw(image, startX, startY, endX, endY, label):
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    return image
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5:"Sad", 6:"Surprise"}


def detect(image):
    global angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count
    #image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    image = imutils.resize(image, width=400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape[:2]
    print(w,h)
    print("[INFO] loading model...")
    prototxt = 'deploy.prototxt'
    if (os.path.isfile(prototxt)):
        print("ok")
    else:
        print('[FILE NOT FOUND] prototxt is not found')
        return
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    if os.path.isfile(model):
        print("ok")
    else:
        print('[FILE NOT FOUND] model not found')
        return

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    padding = 20
    emo = load_model('my_model.h5',compile=False, custom_objects={'BatchNormalization': BatchNormalization})
    #print(emo.summary())
    #emo = load_model('my_model.h5',compile=False)
    image = imutils.resize(image, width=400)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    # genderList = ['Male', 'Female']
    sX = 0
    sY = 0
    eX = 0
    eY = 0
    lb = ''
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX,startY,endX,endY)
            (sX,sY,eX,eY) = (startX, startY, endX, endY)
            # draw the bounding box of the face along with the associated probability
            # text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.putText(image, text, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            t = time.time()
            roi_gray_frame = gray[sY:sY + h, sX:sX + w]
            if(len(roi_gray_frame)>0):
                cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), 0)

                # if len(cropped_img) > 0:
                #     face_front += 1

                # predict the emotions
                #emotion_prediction = emo.
                emotion_prediction = emo.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(emotion_prediction))
                # Calculate text size to determine placement
                text_size = cv2.getTextSize(emotion_dict[maxindex], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_width, text_height = text_size[0], text_size[1]

                # Calculate text position
                text_x = startX + int((endX - startX - text_width) / 2)
                text_y = startY - 10  # Above the bounding box

                # Draw the predicted emotion text on the image
                cv2.putText(image, emotion_dict[maxindex], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                lb = emotion_dict[maxindex]
                if emotion_dict[maxindex] == 'Angry':
                    angry_count += 1
                elif emotion_dict[maxindex] == 'Sad':
                    sad_count += 1
                elif emotion_dict[maxindex] == 'Neutral':
                    neutral_count = neutral_count + 1
                elif emotion_dict[maxindex] == 'Fear':
                    fear_count += 1
                elif emotion_dict[maxindex] == 'Surprise':
                    surprise_count += 1
                elif emotion_dict[maxindex] == 'Disgust':
                    disgust_count += 1
                elif emotion_dict[maxindex] == 'Happy':
                    happy_count += 1
    return image,sX,sY,eX,eY,lb
def detectImage():
    global angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count
    path = 'image.jpeg'
    if os.path.isfile(path):
        print("ok")
    else:
        print('FILE NOT FOUND')
        return
    image = cv2.imread(path)
    assert not isinstance(image,type(None)), 'image not found'
    if image is None:
        print('Wrong path:', path)
    else:
        print('[RIGHT] right path:', path)
        print('[INFO] imagesize:', image.shape)
    output, *_ = detect(image)
    cv2.imwrite("image.jpeg", output)
def detectVideo():
  global angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count
  video_path = 'video.mp4'
  output_video_path = 'output_video.mp4'
  cap = cv2.VideoCapture(video_path)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  fps = int(cap.get(5))
  print(frame_width)
  print(frame_height)
  print(fps)
  ret, frame = cap.read()
  if not ret:
      return
  # Detect khuôn mặt trên frame
  frame = imutils.resize(frame, width=400)
  (h, w) = frame.shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
  frame_interval = int(fps * 0.5)  # detect every half second
  frame_count = 0
  image, startX, startY, endX, endY, label = detect(frame)
  # # Ghi frame đã detect vào video output
  out.write(image)
  while True:
      ret, frame = cap.read()
      if not ret:
          break
      # Detect khuôn mặt trên frame
      frame = imutils.resize(frame, width=400)
      (h, w) = frame.shape[:2]
      if frame_count % frame_interval == 0:
          image, sX,sY,eX,eY,lb = detect(frame)
          (startX,startY,endX,endY,label) = (sX,sY,eX,eY,lb)
          # # Ghi frame đã detect vào video output
          out.write(image)
      else:
          image = draw(frame,startX,startY,endX,endY,label)
          out.write(image)
      frame_count += 1
  cv2.destroyAllWindows()
  cap.release()
  out.release()

@app.route('/')
@app.route('/image', methods=['GET', 'POST'])
def image():
    global angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count
    if(request.method == "POST"):
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400  
        file = request.files['file']       
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400        
        if file:
            existing_file_path =  'image.jpeg'
            # If the existing image is a JPEG, overwrite it
            file_path = existing_file_path
            file.save(file_path)        
            detectImage()
            res = cloudinary.uploader.unsigned_upload(open('image.jpeg','rb'), upload_preset='videoApp', resource_type='image')
             # Prepare response with emotion counts and Cloudinary URL
            response = {
                'url': res['url'],
                'emotion_counts': [
                        {
                            'emoFull': 'Neutral',
                            'emo': 'neu',
                            'amount': neutral_count,
                        },
                        {
                            'emoFull': 'Happy',
                            'emo': 'hap',
                            'amount': happy_count,
                        },
                        {
                            'emoFull': 'Sad',
                            'emo': 'sad',
                            'amount': sad_count,
                        },
                        {
                            'emoFull': 'Angry',
                            'emo': 'ang',
                            'amount': angry_count,
                        },
                        {
                            'emoFull': 'Fear',
                            'emo': 'fear',
                            'amount': fear_count,
                        },
                        {
                            'emoFull': 'Disgust',
                            'emo': 'dis',
                            'amount': disgust_count,
                        },
                        {
                            'emoFull': 'Surprise',
                            'emo': 'sup',
                            'amount': surprise_count,
                        }
                    ]
            }
            angry_count = 0
            disgust_count = 0
            fear_count = 0
            happy_count = 0
            neutral_count = 0
            sad_count = 0
            surprise_count = 0
            return jsonify(response)
            #return send_from_directory(app.config['UPLOAD_FOLDER'], 'image.jpeg')
        else:
            return jsonify({"error": "File upload failed"}), 400
    if(request.method == "GET"):
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'image.jpeg')

    #     urllib.request.urlretrieve(bytesOfImage, 'image.jpeg')
    #     #detectImage()
    #     with open('image.jpeg', 'rb') as image_file:
    #         encoded_string = base64.b64encode(image_file.read())
    #     #response = make_response(send_file(file_path,mimetype='image/png'))
    #    # res = cloudinary.uploader.unsigned_upload(open('image.jpeg','rb'), upload_preset='videoApp', resource_type='image')
        
    #     return jsonify({'url': res['url']})

@app.route("/video", methods=['GET', 'POST'])
def video():
    global angry_count, disgust_count, fear_count, happy_count, neutral_count, sad_count, surprise_count
    if(request.method == "POST"):
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400   
        file = request.files['file']       
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400      
        if file:
            #existing_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpeg')
            existing_file_path =  'video.mp4'
            # If the existing image is a JPEG, overwrite it
            file_path = existing_file_path
            file.save(file_path)        
            detectVideo()
            res = cloudinary.uploader.unsigned_upload(open('output_video.mp4','rb'), upload_preset='videoApp', resource_type='video')
            response = {
                'url': res['url'],
                'emotion_counts': [
                        {
                            'emoFull': 'Neutral',
                            'emo': 'neu',
                            'amount': neutral_count,
                        },
                        {
                            'emoFull': 'Happy',
                            'emo': 'hap',
                            'amount': happy_count,
                        },
                        {
                            'emoFull': 'Sad',
                            'emo': 'sad',
                            'amount': sad_count,
                        },
                        {
                            'emoFull': 'Angry',
                            'emo': 'ang',
                            'amount': angry_count,
                        },
                        {
                            'emoFull': 'Fear',
                            'emo': 'fear',
                            'amount': fear_count,
                        },
                        {
                            'emoFull': 'Disgust',
                            'emo': 'dis',
                            'amount': disgust_count,
                        },
                        {
                            'emoFull': 'Surprise',
                            'emo': 'sup',
                            'amount': surprise_count,
                        }
                    ]
            }
            angry_count = 0
            disgust_count = 0
            fear_count = 0
            happy_count = 0
            neutral_count = 0
            sad_count = 0
            surprise_count = 0
            return jsonify(response)
            #return send_from_directory(app.config['UPLOAD_FOLDER'], 'image.jpeg')
        else:
            return jsonify({"error": "File upload failed"}), 400

    # if(request.method == "POST"):
    #     bytesOfVideo = request.get_data().decode('utf-8')
    #     print(bytesOfVideo)
    #     # with open('video.mp4', 'wb') as out:
    #     #     out.write(base64.b64decode(bytesOfVideo))
    #     # video_url = cloudinary.api.resource(bytesOfVideo)['url']
    #     # print(video_url)

    #     urllib.request.urlretrieve(bytesOfVideo, 'video.mp4')
    #     #detectVideo()
    #     with open("output_video.mp4", "rb") as videoFile:
    #         text = base64.b64encode(videoFile.read())
    #         #print(text)
    #     #res = cloudinary.uploader.unsigned_upload(open('output_video.mp4','rb'), upload_preset='videoApp', resource_type='video')
        
    #     return jsonify({'url': res['url']})

