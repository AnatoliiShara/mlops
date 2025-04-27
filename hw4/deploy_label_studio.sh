#!/bin/bash
# Перейти до hw4
cd ~/Documents/mlops_projector/hw4

# Створити директорію для даних Label Studio
mkdir -p data/label-studio-data

# Змінити права доступу
chmod -R 777 data/label-studio-data

# Запустити Label Studio через Docker
docker run -it -p 8080:8080 \
  -v $(pwd)/data/label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest

# Повідомлення про запуск
echo "Label Studio is running at http://localhost:8080"