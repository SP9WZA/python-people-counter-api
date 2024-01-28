from flask import Flask
from flask_restful import Resource, Api
import cv2

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread("przystanek.jpg")
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        # print(type(img))
        # print(img.shape)
        return {'count': len(boxes)}


api.add_resource(HelloWorld, '/test')
api.add_resource(PeopleCounter, '/count')

if __name__ == '__main__':
    app.run(debug=True, port=8000)


