import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import environ
import sys
from google.cloud import storage

assert insightface.__version__>='0.7'

bucket = environ.get('BUCKET')
source_face_index = int(environ.get('SOURCE_FACE_INDEX'))
image_file = environ.get('IMAGE_FILE')

def download_blob(bucket_name, blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    print(
        f"Blob {blob_name} from bucket {bucket_name} downloading to {destination_file_name}."
    )
    sys.stdout.flush()

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    destination_dir = os.path.dirname(destination_file_name)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {blob_name} downloaded to {destination_file_name}."
    )
    sys.stdout.flush()

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}."
    )

def run():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    download_blob(bucket, image_file, image_file)

    img = cv2.imread(image_file)
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    source_face = faces[source_face_index]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    name, ext = image_file.split(".")
    out_file = f"{name}_swapped.{ext}"
    print(f"Writing the swapped file to ${out_file}")

    cv2.imwrite(out_file, res)
    upload_blob(bucket, out_file, out_file)

run()
