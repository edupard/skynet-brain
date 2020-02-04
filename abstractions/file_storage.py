from google.cloud import storage
import os

_PROJECT_NAME = 'skynet-1984'
_BUCKET_NAME = f'{_PROJECT_NAME}-data'

def put_file(local_file_path, remote_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(_BUCKET_NAME)
    blob = bucket.blob(remote_file_name)
    blob.upload_from_filename(local_file_path)

def get_file(remote_file_name, local_file_name):
    dir = os.path.dirname(local_file_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(_BUCKET_NAME)
    blob = bucket.get_blob(remote_file_name)
    if blob is None:
        raise Exception(f"Can not find gs://{_BUCKET_NAME}/{remote_file_name}")
    blob.download_to_filename(local_file_name)

def remove_file(remote_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(_BUCKET_NAME)
    blob = bucket.get_blob(remote_file_name)
    if blob is not None:
        blob.delete()