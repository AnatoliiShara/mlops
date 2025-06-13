.PHONY: help build run test deploy clean

DOCKER_IMAGE = book-recommender:latest
NAMESPACE = ml-hw12

help:
	@echo "Available commands:"
	@echo "  make build       - Build Docker image"
	@echo "  make run         - Run locally with docker-compose"
	@echo "  make deploy      - Deploy to Kubernetes"
	@echo "  make test-hpa    - Test HPA with load"
	@echo "  make clean       - Clean up resources"

build:
	docker build -t $(DOCKER_IMAGE) .

run:
	docker-compose up

deploy:
	# Install metrics-server if not exists
	kubectl get deployment metrics-server -n kube-system || \
		kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	
	# Apply Kubernetes manifests
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml
	
	# Wait for deployment
	kubectl wait --for=condition=available --timeout=300s deployment/book-recommender -n $(NAMESPACE)

test-hpa:
	@echo "Starting load test to trigger HPA..."
	kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -n $(NAMESPACE) -- \
		/bin/sh -c "while sleep 0.01; do wget -q -O- http://book-recommender-service/recommend -d '{\"query\":\"test\"}'; done"

port-forward:
	kubectl port-forward -n $(NAMESPACE) svc/book-recommender-service 8000:80

logs:
	kubectl logs -f -n $(NAMESPACE) -l app=book-recommender

clean:
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	docker-compose down