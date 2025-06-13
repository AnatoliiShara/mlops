import pandas as pd
import nni
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

# 1. Отримання гіперпараметрів із NNI
params = nni.get_next_parameter()
n_factors = params.get("n_factors", 10)  # Значення за замовчуванням
lr_all = params.get("lr_all", 0.005)
reg_all = params.get("reg_all", 0.02)

# 2. Завантаження даних для Collaborative Filtering
ratings_data = pd.read_csv("data/user_book_ratings.csv")
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings_data[["user_id", "book_id", "rating"]], reader)

# 3. Розділення на train/test
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# 4. Тренування SVD
model = SVD(
    n_factors=n_factors,
    n_epochs=20,  # Залишимо фіксованим
    lr_all=lr_all,
    reg_all=reg_all
)
model.fit(trainset)

# 5. Оцінка
predictions = model.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

# 6. Звітність результатів у NNI (оптимізуємо RMSE)
nni.report_final_result(rmse)

# 7. Збереження моделі (опціонально, для найкращого запуску)
with open("models/svd_model_nni.pkl", "wb") as f:
    pickle.dump(model, f)