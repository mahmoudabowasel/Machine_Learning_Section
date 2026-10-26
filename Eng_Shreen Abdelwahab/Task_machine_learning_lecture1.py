# Mahmoud Kamal Abu Wasel Awad
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
)


def classification_metrics_example():
    """Reproduce the lecture's cancer-prediction confusion-matrix example."""
    print("\n" + "=" * 65)
    print("1) CLASSIFICATION METRICS — LECTURE EXAMPLE")
    print("=" * 65)

    y_true = np.array([1] * 100 + [0] * 900)
    y_pred = np.array([1] * 90 + [0] * 10 + [1] * 5 + [0] * 895)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"TP (True Positive)  = {tp}")
    print(f"FN (False Negative) = {fn}")
    print(f"FP (False Positive) = {fp}")
    print(f"TN (True Negative)  = {tn}")
    print(f"Total observations  = {len(y_true)}")

    print("\nConfusion Matrix (rows = actual [0, 1], columns = predicted [0, 1]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nMetrics:")
    print(f"Accuracy  = {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Precision = {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Recall    = {recall:.4f} ({recall * 100:.1f}%)")
    print(f"F1-Score  = {f1:.4f} ({f1 * 100:.1f}%)")


def supervised_learning_example():
    """Train a classifier on labeled Iris flower measurements."""
    print("\n" + "=" * 65)
    print("2) SUPERVISED LEARNING — IRIS CLASSIFICATION")
    print("=" * 65)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.20,
        random_state=42,
        stratify=iris.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Accuracy        : {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            zero_division=0,
        )
    )

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    predicted_class = model.predict(sample)[0]
    print("New flower measurements:", sample[0])
    print("Predicted species:", iris.target_names[predicted_class])


def unsupervised_learning_example():
    """Use K-Means on flower measurements without using species labels."""
    print("\n" + "=" * 65)
    print("3) UNSUPERVISED LEARNING — K-MEANS CLUSTERING")
    print("=" * 65)

    iris = load_iris()
    X = iris.data

    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=3, random_state=42, n_init=10),
    )
    clusters = model.fit_predict(X)

    print("K-Means grouped the unlabeled samples into 3 clusters.")
    print("Number of samples in each cluster:")
    cluster_ids, counts = np.unique(clusters, return_counts=True)
    for cluster_id, count in zip(cluster_ids, counts):
        print(f"  Cluster {cluster_id}: {count} samples")


def regression_metrics_example():
    """Calculate the regression metrics covered in the lecture."""
    print("\n" + "=" * 65)
    print("4) REGRESSION METRICS — MSE, RMSE, MAE")
    print("=" * 65)

    y_actual = np.array([3.0, -0.5, 2.0, 7.0])
    y_predicted = np.array([2.5, 0.0, 2.0, 8.0])

    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_predicted)

    print("Actual values    :", y_actual)
    print("Predicted values :", y_predicted)
    print(f"MSE  = {mse:.4f}  (squares large errors more heavily)")
    print(f"RMSE = {rmse:.4f}  (same unit as the target)")
    print(f"MAE  = {mae:.4f}  (average absolute error)")


def reinforcement_learning_example():
    """A small Q-learning example: move along states 0..4 to reach goal state 4."""
    print("\n" + "=" * 65)
    print("5) REINFORCEMENT LEARNING — SIMPLE Q-LEARNING")
    print("=" * 65)

    n_states = 5
    actions = [0, 1]
    goal_state = 4

    q_table = np.zeros((n_states, len(actions)))
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_rate = 0.2
    episodes = 500
    rng = np.random.default_rng(42)

    for _ in range(episodes):
        state = 0
        for _step in range(20):
            if rng.random() < exploration_rate:
                action = int(rng.choice(actions))
            else:
                action = int(np.argmax(q_table[state]))

            if action == 0:
                next_state = max(0, state - 1)
            else:
                next_state = min(goal_state, state + 1)

            reached_goal = next_state == goal_state
            reward = 10 if reached_goal else -1

            best_future_value = 0 if reached_goal else np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (
                reward
                + discount_factor * best_future_value
                - q_table[state, action]
            )

            state = next_state
            if reached_goal:
                break

    print("Learned Q-table (rows = states 0..4, columns = [left, right]):")
    print(np.round(q_table, 2))

    state = 0
    route = [state]
    for _ in range(10):
        action = int(np.argmax(q_table[state]))
        state = max(0, state - 1) if action == 0 else min(goal_state, state + 1)
        route.append(state)
        if state == goal_state:
            break

    print("Route chosen by the learned policy:", " -> ".join(map(str, route)))
    print("Goal reached:", state == goal_state)


def main():
    print("MACHINE LEARNING — PRACTICAL SESSION")
    classification_metrics_example()
    supervised_learning_example()
    unsupervised_learning_example()
    regression_metrics_example()
    reinforcement_learning_example()
    print("\nDone. Compare the printed results with the lecture formulas.")


if __name__ == "__main__":
    main()
