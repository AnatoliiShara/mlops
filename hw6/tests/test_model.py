from rec_sys.pipeline import train_svd


def test_svd_training_fast(ratings_df):
    small = ratings_df.sample(frac=0.1, random_state=0)
    model, rmse = train_svd(
        small,
        dict(n_factors=8, n_epochs=8, lr_all=0.02, reg_all=0.02),
        seed=0,
    )
    assert rmse < 2.0          # лояльніший поріг
    assert hasattr(model, "predict")
