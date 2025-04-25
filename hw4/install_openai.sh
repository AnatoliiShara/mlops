#!/bin/bash
# Перейти до hw4
cd ~/Documents/mlops_projector/hw4

# Активувати віртуальне середовище
source ../venv/bin/activate

# Встановити бібліотеку openai
pip install openai

# Перевірити встановлення
pip show openai