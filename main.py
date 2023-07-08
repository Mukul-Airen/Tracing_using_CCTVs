from flask import Flask, render_template, Response, request
from Detector import *
from FaceRecognition import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen1(Detector):
    #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    #modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    #modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz"
    classFile = "coco.names"
    Detector.readClasses(classFile)
    Detector.downloadModel(modelURL)
    Detector.loadModel()
    while True:
        frame = Detector.predictVideo()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen2(faceRecogination):
    while True:
        frame = faceRecogination.FaceReco()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen1(Detector()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/face')
def face():
    return Response(gen2(FaceRecognition()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_face')
def detect_face():
    return render_template('facerec.html')

@app.route('/setting')
def setting():
    return render_template('setting.html')

@app.route('/submit-form', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':
        a_sid = request.form['a_sid']
        auth_token = request.form['auth_token']
        twilio_no = request.form['twilio_no']
        ph_no = request.form['ph_no']

        with open('keys.py', 'w') as f:
            f.write(f"account_sid = '{a_sid}'\n")
            f.write(f"auth_token = '{auth_token}'\n")
            f.write(f"twilio_number = '{twilio_no}'\n")
            f.write(f"my_phone_number = '{ph_no}'")

        return render_template('submit.html')

    return render_template('setting.html')

if __name__ == '__main__':
    app.run(debug=True)
