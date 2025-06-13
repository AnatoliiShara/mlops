from minio import Minio
from minio.error import S3Error
import os

class MinioClient:
    def __init__(self, endpoint="localhost:9000", access_key="admin", secret_key="password", secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = "test-bucket"

    def create_bucket(self):
        """Create a bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket {self.bucket_name} created.")
            else:
                print(f"Bucket {self.bucket_name} already exists.")
        except S3Error as e:
            print(f"Error creating bucket: {e}")
            raise

    def upload_file(self, file_path, object_name):
        """Upload a file to the bucket (Create)."""
        try:
            self.client.fput_object(self.bucket_name, object_name, file_path)
            print(f"File {file_path} uploaded as {object_name}.")
        except S3Error as e:
            print(f"Error uploading file: {e}")
            raise

    def download_file(self, object_name, file_path):
        """Download a file from the bucket (Read)."""
        try:
            self.client.fget_object(self.bucket_name, object_name, file_path)
            print(f"File {object_name} downloaded to {file_path}.")
        except S3Error as e:
            print(f"Error downloading file: {e}")
            raise

    def update_file(self, file_path, object_name):
        """Update a file in the bucket (Update)."""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            self.client.fput_object(self.bucket_name, object_name, file_path)
            print(f"File {object_name} updated with {file_path}.")
        except S3Error as e:
            print(f"Error updating file: {e}")
            raise

    def delete_file(self, object_name):
        """Delete a file from the bucket (Delete)."""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            print(f"File {object_name} deleted.")
        except S3Error as e:
            print(f"Error deleting file: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    client = MinioClient()
    client.create_bucket()
    client.upload_file("example.txt", "example.txt")
    client.download_file("example.txt", "downloaded_example.txt")
    client.update_file("example_updated.txt", "example.txt")
    client.delete_file("example.txt")