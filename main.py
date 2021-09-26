import tensorflow as tf
import debugpy
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, make_response
from cv2 import cv2
from flask import send_from_directory, app
from werkzeug.utils import safe_join
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
#debugpy.listen(5678)
#debugpy.wait_for_client()
camera = cv2.VideoCapture(2)
model=keras.models.load_model('sign_language_model.h5')
cm_Plot_labels=['0','1','2','3','4','5','6','7','8','9','A','Aboard','All_Gone','B','Baby','Beside','Book','Bowl','Bridge','C','Camp','Cartridge','D','E','F','Fond','Friend','G','Glove','H','Hang','High','House','How_many','I','J','K','L','M','Man','Marry','Meat','Medal','Middle','Money','Mother','N','O','Opposite','P','Prisoner','Q','R','Ring','Rose','S','See','Short','Superior','T','Tabacoo','Thick','Thin','U','V','W','Watch','Write','X','Y','You','Z']

def process_frame(encodedImage):
    frame=cv2.flip(encodedImage, 1)
    cv2image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(cv2image, (224, 224))
    img = img.reshape((1, 224, 224, 3))
    img = img.astype('float32')/ 255.0
    return img


def gen_frames():
    while True:
        ret, frame = camera.read()
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if ret:
            image1=process_frame(encodedImage)
            predict_result(image1)
            TextImage=cv2.rectangle(encodedImage, (10, 10), (100, 100), (0, 255, 0), 2)
            #TextImage=cv2.putText(encodedImage, resultvalue, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 10)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(TextImage) + b'\r\n\r\n')

def predict_result(image1):
    pred = model.predict(image1)
    resultOfPredication = np.argmax(pred)
    resultvalue = cm_Plot_labels[resultOfPredication]
    print(resultvalue)
    return resultvalue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project.html', methods= ['GET'])
def project(output='demo'):
    gen_frames()
    r1 = predict_result()
    return render_template('project.html', output=r1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result_text', methods= ['GET'])
def result_text():
    gen_frames()
    r = predict_result()
    response = make_response(r, 200)
    response.mimetype = "text/plain"
    return response

@app.route('/<any(css, images, js, fonts):folder>/<path:filename>')
def toplevel_static(folder, filename):
    filename = safe_join(folder, filename)
    cache_timeout = app.get_send_file_max_age(filename)
    return send_from_directory(app.static_folder, filename,
                               cache_timeout=cache_timeout)

@app.route('/<path:htmlfile>.html')
def html_dyanamic1(htmlfile):
    filename = htmlfile + '.html'
    cache_timeout = app.get_send_file_max_age(filename)
    return send_from_directory(app.template_folder, filename,
                               cache_timeout=cache_timeout)

if __name__ == '__main__':
    app.run(debug=True)
