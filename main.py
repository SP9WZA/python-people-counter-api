from flask import Flask, request, render_template, make_response
from flask_restful import Resource, Api
import cv2
import requests

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
        return {'count': len(boxes)}


class PeopleCounterFromWeb(Resource):
    def get(self):
        img_url = request.args.get('url')
        downloaded_img = requests.get(img_url, stream=True)
        with open("current_image", 'wb') as f:
            f.write(downloaded_img.content)
        img = cv2.imread("current_image")
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {'count': len(boxes)}


class PeopleCounterUpload(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('upload_image.html'), 200,
                             headers)

    def post(self):
        file = request.files['file']
        filename = "current_uploaded_file"
        file.save(filename)
        img = cv2.imread("current_uploaded_file")

        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {'count': len(boxes)}


api.add_resource(HelloWorld, '/test')
api.add_resource(PeopleCounter, '/count')
api.add_resource(PeopleCounterFromWeb, '/count_from_web')
api.add_resource(PeopleCounterUpload, '/count_upload')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
