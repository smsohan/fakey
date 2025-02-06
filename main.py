import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from google.cloud import storage
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import environ
import sys

assert insightface.__version__>='0.7'

def download_blob(bucket_name, blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    print(
        f"Blob {blob_name} from bucket {bucket_name} downloading to {destination_file_name}."
    )
    sys.stdout.flush()

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

        # Check if the destination directory exists, create it if not
    destination_dir = os.path.dirname(destination_file_name)
    if destination_dir:  # Check if a directory part exists
        os.makedirs(destination_dir, exist_ok=True) # Create the directory

    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {blob_name} downloaded to {destination_file_name}."
    )
    sys.stdout.flush()

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    # Construct a Cloud Storage client object.
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object from the file you want to upload
    blob = bucket.blob(destination_blob_name)

    # Upload the file to your bucket.
    # You can also use blob.upload_from_file(file_obj) if you have a file-like object.
    blob.upload_from_filename(source_file_name)  # For local files

    print(
        f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}."
    )

def run():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    image_file = "input.jpg"
    download_blob("posts_db", image_file, image_file)

    img = cv2.imread(image_file)
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    source_face = faces[0]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("./t1_swapped.jpg", res)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("./t1_swapped2.jpg", res)

    upload_blob("posts_db", "./t1_swapped.jpg", "t1_swapped.jpg")
    upload_blob("posts_db", "./t1_swapped2.jpg", "t1_swapped2.jpg")


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        run()
        self.send_response(200)  # 200 OK
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Hello, World!</h1></body></html>")

if __name__ == '__main__':
    if environ.get('PORT') is not None:
        server_address = ('', 8080)  # Listen on all interfaces, port 8000
        httpd = HTTPServer(server_address, MyHandler)
        print('Starting server...')
        httpd.serve_forever()
    else:
        run()
