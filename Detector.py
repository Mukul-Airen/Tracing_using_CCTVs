import cv2
import time
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from twilio.rest import Client
import keys
import os
from tensorflow.python.keras.utils.data_utils import get_file

class Detector(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def animal(self, name):
        client = Client(keys.account_sid, keys.auth_token)
        threats = ['cow', 'elephant','bear','horse','sheep','giraffe']
        if name in threats:
            with open('animals.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    f.writelines(f'{name},{dtString}\n')
                    message = client.messages.create(
                        body=name+" Detected in the road by camera 1",
                        from_=keys.twilio_number,
                        to=keys.my_phone_number
                    )
                    print(message.body)

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.colorList = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName,
                 origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(
            self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded successfully...")

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(
            np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                               iou_threshold=threshold, score_threshold=threshold)

        print(bboxIdx)
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex]
                classColor = self.colorList[classIndex]

                self.animal(classLabelText)
                displayText = '{} : {}%'.format(
                    classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (
                    xmin*imW, xmax*imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(
                    xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                              color=classColor, thickness=2)
                cv2.putText(image, displayText, (xmin, ymin-10),
                            cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
        return image

    def predictVideo(self):
        success, image = self.video.read()
        startTime = 0
        threshold = 0.5
        while success:
            currentTime = time.time()
            fps = 1/(currentTime-startTime)
            startTime = currentTime
            bboxImage = self.createBoundingBox(image, threshold)
            cv2.putText(bboxImage, "FPS" + str(int(fps)), (20, 70),
                        cv2.FONT_HERSHEY_PLAIN, 2, (81, 45, 168), 2)
            success, image = self.video.read()
            ret, jpeg = cv2.imencode('.jpg', bboxImage)
            return jpeg.tobytes()
