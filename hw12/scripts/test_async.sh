#!/bin/bash

API_URL="http://localhost:8000"

echo "Testing Async Book Recommender API..."

# Health check
echo -e "\n1. Health Check:"
curl -s $API_URL/health | jq .

# Submit job
echo -e "\n2. Submitting recommendation job:"
JOB_RESPONSE=$(curl -s -X POST $API_URL/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "психологічний трилер",
    "top_k": 5
  }')

echo $JOB_RESPONSE | jq .

# Extract job_id
JOB_ID=$(echo $JOB_RESPONSE | jq -r .job_id)
echo "Job ID: $JOB_ID"

# Check job status
echo -e "\n3. Checking job status:"
sleep 2
curl -s $API_URL/job/$JOB_ID | jq .

# Wait for completion
echo -e "\n4. Waiting for job completion..."
for i in {1..10}; do
  sleep 2
  STATUS=$(curl -s $API_URL/job/$JOB_ID | jq -r .status)
  echo "Status: $STATUS"
  if [ "$STATUS" == "completed" ]; then
    break
  fi
done

# Get final result
echo -e "\n5. Final result:"
curl -s $API_URL/job/$JOB_ID | jq .

# Check queue status
echo -e "\n6. Queue status:"
curl -s $API_URL/queue/status | jq .

# Check metrics
echo -e "\n7. Prometheus metrics:"
curl -s $API_URL/metrics | grep -E "(jobs_submitted_total|jobs_by_status|queue_length)"