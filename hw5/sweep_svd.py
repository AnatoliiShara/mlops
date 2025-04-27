import pandas as pd
import wandb
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

# Ініціалізація W&B (буде викликано W&B Sweeps)
wandb.init()

# 1. Завантаження даних для Collaborative Filtering
ratings_data = pd.read_csv("data/user_book_ratings.csv")
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings_data[["user_id", "book_id", "rating"]], reader)

# 2. Розділення на train/test
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. Отримання гіперпараметрів із W&B Sweeps
config = wandb.config

# 4. Тренування SVD
model = SVD(
    n_factors=config.n_factors,
    n_epochs=20,  # Залишимо фіксованим
    lr_all=config.lr_all,
    reg_all=config.reg_all
)
model.fit(trainset)

# 5. Оцінка
predictions = model.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

# 6. Логування метрик у W&B
wandb.log({"RMSE": rmse, "MAE": mae})

# 7. Збереження моделі (опціонально, для найкращого запуску)
with open("models/svd_model_sweep.pkl", "wb") as f:
    pickle.dump(model, f)

artifact = wandb.Artifact("svd-model-sweep", type="model")
artifact.add_file("models/svd_model_sweep.pkl")
wandb.log_artifact(artifact)

# Завершення W&B (не потрібно викликати wandb.finish(), W&B Sweeps зробить це автоматично)