import ssl
import urllib.request
import certifi
import cv2
from cvzone.FaceDetectionModule import FaceDetector

ssl_context = ssl.create_default_context(cafile=certifi.where())
https_handler = urllib.request.HTTPSHandler(context=ssl_context)
urllib.request.install_opener(urllib.request.build_opener(https_handler))

detector = FaceDetector()


cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    img, bboxes = detector.findFaces(img)
    if bboxes:
        center = bboxes[0]['center']
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
