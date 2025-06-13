import pytest
import os
from minio_client import MinioClient

@pytest.fixture
def minio_client():
    client = MinioClient()
    client.create_bucket()
    return client

@pytest.fixture
def test_file(tmp_path):
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("Test content")
    return file_path

def test_create_bucket(minio_client):
    assert minio_client.client.bucket_exists("test-bucket")

def test_upload_file(minio_client, test_file):
    minio_client.upload_file(str(test_file), "test.txt")
    assert minio_client.client.stat_object("test-bucket", "test.txt")

def test_download_file(minio_client, test_file, tmp_path):
    minio_client.upload_file(str(test_file), "test.txt")
    download_path = tmp_path / "downloaded.txt"
    minio_client.download_file("test.txt", str(download_path))
    assert os.path.exists(download_path)
    with open(download_path, "r") as f:
        assert f.read() == "Test content"

def test_update_file(minio_client, test_file, tmp_path):
    minio_client.upload_file(str(test_file), "test.txt")
    updated_file = tmp_path / "updated.txt"
    with open(updated_file, "w") as f:
        f.write("Updated content")
    minio_client.update_file(str(updated_file), "test.txt")
    download_path = tmp_path / "downloaded_updated.txt"
    minio_client.download_file("test.txt", str(download_path))
    with open(download_path, "r") as f:
        assert f.read() == "Updated content"

def test_delete_file(minio_client, test_file):
    minio_client.upload_file(str(test_file), "test.txt")
    minio_client.delete_file("test.txt")
    with pytest.raises(Exception):
        minio_client.client.stat_object("test-bucket", "test.txt")