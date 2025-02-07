import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import environ
import sys
import uuid

assert insightface.__version__>='0.7'

bucket = environ.get('BUCKET')

def run(image_file="input.jpg"):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    img = cv2.imread(image_file)
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    source_face = faces[0]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    return res

class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            filename = f"image_{uuid.uuid4()}.jpg"
            out_file = f"out_{uuid.uuid4()}.jpg"
            filepath = os.path.join(".", filename)

            with open(filepath, "wb") as f:
                f.write(post_data)
            print(f"Image saved to: {filepath}")

            result = run(filename)
            is_success, encoded_image = cv2.imencode('.jpg', result)
            if is_success:
                image_bytes = encoded_image.tobytes()

                self.send_response(200)
                self.send_header('Content-type', "image/jpg")
                self.send_header('Content-Length', len(image_bytes))
                self.end_headers()
                self.wfile.write(image_bytes)
            else:
                raise Exception("Sorry, failed to convert to jpg")
        except Exception as e:
            print(f"Error processing or saving image: {e}")
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error processing or saving image: {e}")

if __name__ == '__main__':
    if environ.get('PORT') is not None:
        server_address = ('', 8080)  # Listen on all interfaces, port 8000
        httpd = HTTPServer(server_address, MyHandler)
        print('Starting server...')
        httpd.serve_forever()
