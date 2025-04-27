MinIO Deployment Instructions
This document provides step-by-step instructions for deploying MinIO in three configurations: locally, using Docker, and in Kubernetes (K8S).
Prerequisites

Ubuntu (or similar Linux distribution)
Docker and docker-compose installed (for Docker deployment)
Minikube, kubectl, and Helm installed (for Kubernetes deployment)
Internet access for downloading MinIO binaries and images

1. Local Deployment
Steps

Download MinIO binary:
wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/Documents/mlops_projector/hw3/minio_deployment/local/minio
chmod +x ~/Documents/mlops_projector/hw3/minio_deployment/local/minio


Create a script to run MinIO: Save the following as run_minio.sh:
#!/bin/bash
export MINIO_ROOT_USER=admin
export MINIO_ROOT_PASSWORD=password
mkdir -p ~/minio-data
~/Documents/mlops_projector/hw3/minio_deployment/local/minio server ~/minio-data --console-address ":9001"


Run MinIO:
chmod +x run_minio.sh
./run_minio.sh


Access MinIO:

Web UI: Open http://localhost:9001 in a browser.
Login with admin:password.
API endpoint: http://localhost:9000.



2. Docker Deployment
Steps

Create a docker-compose.yml file: Save the following in minio_deployment/docker/docker-compose.yml:
version: '3.8'
services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD=password
    volumes:
      - minio-data:/data
    command: server /data --console-address ":9001"
volumes:
  minio-data:


Run MinIO:
cd ~/Documents/mlops_projector/hw3/minio_deployment/docker
docker-compose up -d


Access MinIO:

Web UI: http://localhost:9001
Login: admin:password
API: http://localhost:9000


Stop MinIO:
docker-compose down



3. Kubernetes (K8S) Deployment
Steps

Start Minikube (if not already running):
minikube start


Add MinIO Helm repository:
helm repo add minio https://helm.min.io/
helm repo update


Create Kubernetes manifests: Save the following as minio-deployment.yaml:
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        env:
        - name: MINIO_ROOT_USER
          value: "admin"
        - name: MINIO_ROOT_PASSWORD
          value: "password"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        emptyDir: {}

Save the following as minio-service.yaml:
apiVersion: v1
kind: Service
metadata:
  name: minio-service
spec:
  selector:
    app: minio
  ports:
  - name: api
    port: 9000
    targetPort: 9000
  - name: console
    port: 9001
    targetPort: 9001
  type: NodePort


Apply manifests:
kubectl apply -f minio-deployment.yaml
kubectl apply -f minio-service.yaml


Access MinIO:

Get the NodePort:
minikube service minio-service --url


Use the provided URL for the console.

Login: admin:password.



Clean up (optional):
kubectl delete -f minio-deployment.yaml
kubectl delete -f minio-service.yaml



Troubleshooting

Local: Ensure port 9000/9001 are free (sudo netstat -tulnp | grep 9000).
Docker: Check container logs (docker-compose logs minio).
Kubernetes: Use kubectl logs and k9s for debugging.

