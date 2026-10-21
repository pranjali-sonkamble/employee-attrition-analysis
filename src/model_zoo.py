from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def get_models():
    """Return the project's fixed baseline models and scaling choices."""
    return {
        "Dummy": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }
