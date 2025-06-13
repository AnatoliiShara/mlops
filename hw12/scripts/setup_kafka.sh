#!/bin/bash
set -e

echo "Setting up Kafka infrastructure..."

# Start local Kafka cluster
docker-compose -f docker-compose.kafka.yaml up -d

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 30

# Create topic
docker exec kafka kafka-topics \
  --create \
  --topic book-recommendations \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=86400000 \
  --config compression.type=gzip

# List topics
echo "Created topics:"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo "Kafka setup complete!"