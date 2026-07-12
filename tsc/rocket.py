# This code has been modified to suit thesis needs: BS, DS, FS, GS, SD exercises
# Save to txt file command with date stamp:
# python -m tsc.rocket --rocket_config tsc/rocket_config 2>&1 | tee "results_$((Get-Date).ToString('yyyy-MM-dd_HH-mm')).txt"
# Run the program:
# python -m tsc.rocket --rocket_config tsc/rocket_config
# Run the code with alpha log output:
# python -m tsc.rocket --rocket_config tsc/rocket_config 2>&1 | tee "results_$((Get-Date).ToString('yyyy-MM-dd_HH-mm')).txt"
import argparse
import configparser
import os
import sys
import logging
import time
import json
import pandas as pd
import numpy as np
import joblib

from configobj import ConfigObj
from sklearn import metrics
from sklearn.linear_model import RidgeClassifierCV
# from sktime.transformations.panel.rocket import Rocket
from sktime.transformers.series_as_features.rocket import Rocket
# from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.utils.load_data import load_from_tsfile_to_dataframe

from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

# THE NUCLEAR FIX: Forcefully strip out any sneaky logging settings from other libraries
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Force the stream to standard output
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

FILE_NAME_X = "{}_{}_X"
FILE_NAME_Y = "{}_{}_Y"
FILE_NAME_PID = "{}_{}_pid"


def parse_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def max_len_of_nested_df(X):
    if X is None:
        return 0
    return int(X.applymap(len).to_numpy().max())


def pad_series_to_len(s, target_len: int):
    """Pad a series to target_len and return as pd.Series (CRITICAL FIX)."""
    if s is None:
        return s

    if hasattr(s, "to_numpy"):
        arr = s.to_numpy()
    else:
        arr = np.asarray(s)

    n = len(arr)
    if n == target_len:
        return pd.Series(arr)
    if n > target_len:
        return pd.Series(arr[:target_len])

    pad_width = target_len - n
    if n == 0:
        return pd.Series(np.zeros(target_len, dtype=float))

    last_val = arr[-1]
    pad_vals = np.full(pad_width, last_val, dtype=arr.dtype)
    return pd.Series(np.concatenate([arr, pad_vals], axis=0))


def pad_nested_df_to_len(X, target_len: int):
    return X.applymap(lambda s: pad_series_to_len(s, target_len))


def read_dataset(path, data_type):
    train_path = os.path.join(path, FILE_NAME_X.format("TRAIN", data_type) + ".ts")
    x_train, y_train = load_from_tsfile_to_dataframe(train_path)

    lens = x_train.applymap(lambda s: len(s) if s is not None else -1)
    logger.info(f"[TRAIN] Shape: {x_train.shape}, Max Len: {lens.to_numpy().max()}")

    test_path = os.path.join(path, FILE_NAME_X.format("TEST", data_type) + ".ts")
    x_test, y_test = load_from_tsfile_to_dataframe(test_path)

    test_pid = np.arange(len(y_test))
    train_pid = np.arange(len(y_train))

    x_val, y_val = None, None
    try:
        val_path = os.path.join(path, FILE_NAME_X.format("VAL", data_type) + ".ts")
        if os.path.exists(val_path):
            x_val, y_val = load_from_tsfile_to_dataframe(val_path)
    except Exception:
        pass

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


class RocketTransformerClassifier:
    def __init__(self, exercise, seed):
        self.exercise = exercise
        self.seed = int(seed)
        self.rocket = None
        self.max_len_ = None
        self.classifier = None
        self.train_acc_ = None

    @timeit
    def fit_rocket(self, x_train, y_train, kernels=10000, target_len=None):  # 10K KERNELS
        if target_len is None:
            target_len = max_len_of_nested_df(x_train)

        self.max_len_ = int(target_len)
        logger.info(f"Padding to length: {self.max_len_}")

        x_train_padded = pad_nested_df_to_len(x_train, self.max_len_)

        # normalise=True forces ROCKET to scale the skeleton data automatically
        self.rocket = Rocket(num_kernels=kernels, normalise=True, random_state=self.seed)  # normalization Z SCORE
        # -----------------------

        self.rocket.fit(x_train_padded)

        x_train_transform = self.rocket.transform(x_train_padded)
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(x_train_transform, y_train)

        best_alpha = float(self.classifier.alpha_)
        logger.info(f"[RIDGE] selected_alpha={best_alpha}")

        train_preds = self.classifier.predict(x_train_transform)
        self.train_acc_ = metrics.accuracy_score(y_train, train_preds)
        logger.info(f"[TRAIN] Accuracy: {self.train_acc_}")

    @timeit
    def predict_rocket(self, x_test, y_test):
        x_test_padded = pad_nested_df_to_len(x_test, self.max_len_)
        x_test_transform = self.rocket.transform(x_test_padded)
        predictions = self.classifier.predict(x_test_transform)

        labels = list(np.sort(np.unique(y_test)))
        cm = metrics.confusion_matrix(y_test, predictions)
        rep_str = metrics.classification_report(y_test, predictions, zero_division=0)
        rep_dict = metrics.classification_report(y_test, predictions, output_dict=True, zero_division=0)
        acc = metrics.accuracy_score(y_test, predictions)

        logger.info(f"[TEST] Accuracy: {acc}")
        logger.info(f"\n[TEST] Classification Report:\n{rep_str}")

        return {
            "test": {
                "pred": predictions, "cm": cm, "report_str": rep_str,
                "accuracy": acc, "report_dict": rep_dict, "labels": labels
            }
        }


def extract_and_log_participant_profile(x_dataframe, algorithm, exercise):
    """
    Extracts the first frame coordinates of the dataset to calculate
    the Torso-to-Limb Skeletal Ratio for participant profiling.
    """
    try:
        if x_dataframe is None or len(x_dataframe) == 0:
            return None, "Unknown"

        # Grab the very first repetition sample available in the dataframe
        first_rep_series = x_dataframe.iloc[0]

        # Extract first frame [index 0] of the parallel series channels
        ls_x, ls_y = first_rep_series.iloc[0][0], first_rep_series.iloc[1][0]  # Left Shoulder
        rs_x, rs_y = first_rep_series.iloc[2][0], first_rep_series.iloc[3][0]  # Right Shoulder
        lh_x, lh_y = first_rep_series.iloc[4][0], first_rep_series.iloc[5][0]  # Left Hip
        rh_x, rh_y = first_rep_series.iloc[6][0], first_rep_series.iloc[7][0]  # Right Hip
        la_x, la_y = first_rep_series.iloc[8][0], first_rep_series.iloc[9][0]  # Left Ankle
        ra_x, ra_y = first_rep_series.iloc[10][0], first_rep_series.iloc[11][0]  # Right Ankle

        # Calculate Midpoints
        mid_shoulder = np.array([(ls_x + rs_x) / 2, (ls_y + rs_y) / 2])
        mid_hip = np.array([(lh_x + rh_x) / 2, (lh_y + rh_y) / 2])
        mid_ankle = np.array([(la_x + ra_x) / 2, (la_y + ra_y) / 2])

        # Calculate Lengths using Euclidean Distance Formula
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        lower_limb_length = np.linalg.norm(mid_hip - mid_ankle)

        # Safety check to avoid division by zero
        if lower_limb_length == 0:
            return None, "Unknown"

        # Compute Skeletal Frame Aspect Ratio
        skeletal_ratio = torso_length / lower_limb_length

        # Categorize body profile based on structural distribution thresholds
        if skeletal_ratio > 0.85:
            profile_type = "Long-Torso"
        elif skeletal_ratio < 0.75:
            profile_type = "Long-Limb"
        else:
            profile_type = "Symmetrical"

        logger.info(
            f"[PROFILE MANUSCRIPT LOG] Algorithm: {algorithm} | Exercise: {exercise} | Torso-to-Limb Ratio: {skeletal_ratio:.2f} | Profile Category: {profile_type}")
        return skeletal_ratio, profile_type

    except Exception as profile_error:
        logger.warning(f"Could not extract structural profile from dataframe matrix: {profile_error}")
        return None, "Unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    base_path = os.getcwd()
    seed_values = parse_list(rocket_config["SEED_VALUES"])
    exercises = parse_list(rocket_config["EXERCISE"])
    algorithms = parse_list(rocket_config.get("POSE_MODELS", rocket_config.get("ALGORITHMS", ["openpose"])))

    input_data_root = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])
    output_path = rocket_config["OUTPUT_PATH"]
    data_type = rocket_config["DATA_TYPE"]

    output_rocket_root = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_rocket_root)

    run_rows = []

    for algorithm in algorithms:
        logger.info(f"=== PROCESSING ALGORITHM: {algorithm} ===")
        input_data_path = os.path.join(input_data_root, algorithm)
        output_results_root = os.path.join(output_rocket_root, algorithm)
        create_directory_if_not_exists(output_results_root)

        for exercise in exercises:
            logger.info(f"--- Exercise: {exercise} ---")
            exercise_out = os.path.join(output_results_root, exercise)
            create_directory_if_not_exists(exercise_out)

            for seed_value in seed_values:
                input_path_combined = os.path.join(input_data_path, exercise, seed_value)
                train_file = os.path.join(input_path_combined, FILE_NAME_X.format("TRAIN", data_type) + ".ts")
                test_file = os.path.join(input_path_combined, FILE_NAME_X.format("TEST", data_type) + ".ts")

                if not (os.path.exists(train_file) and os.path.exists(test_file)):
                    logger.warning(f"Files not found for {algorithm}/{exercise}. Skipping.")
                    continue

                try:
                    # Read Data
                    x_train, y_train, x_test, y_test, x_val, y_val, _, _ = read_dataset(input_path_combined, data_type)

                    # ======================================================================
                    # 🚀 REVISION LOGIC: PRINT DATA FOR MANUSCRIPT TABLE 3.1
                    # ======================================================================
                    extract_and_log_participant_profile(x_train, algorithm, exercise)
                    # ======================================================================

                    global_max_len = max(
                        max_len_of_nested_df(x_train),
                        max_len_of_nested_df(x_test),
                        max_len_of_nested_df(x_val)
                    )
                    if global_max_len == 0: continue

                    # Fit & Predict on Test Set
                    rocket = RocketTransformerClassifier(exercise, seed_value)
                    rocket.fit_rocket(x_train, y_train, target_len=global_max_len)
                    res = rocket.predict_rocket(x_test, y_test)

                    # Prepare Test Metrics
                    rep = res["test"]["report_dict"]

                    # Predict on Validation Set & Calculate Latency
                    val_acc = "N/A"
                    val_latency_ms = 0.0

                    if x_val is not None and y_val is not None and len(y_val) > 0:
                        try:
                            # --- START INFERENCE TIMER (VALIDATION PHASE) ---
                            start_time_val = time.time()

                            x_val_padded = pad_nested_df_to_len(x_val, rocket.max_len_)
                            x_val_transform = rocket.rocket.transform(x_val_padded)
                            val_pred = rocket.classifier.predict(x_val_transform)

                            # --- END INFERENCE TIMER (VALIDATION PHASE) ---
                            end_time_val = time.time()

                            # Calculate average latency per repetition in milliseconds (ms)
                            num_val_samples = len(x_val)
                            total_val_latency_ms = (end_time_val - start_time_val) * 1000
                            val_latency_ms = total_val_latency_ms / num_val_samples if num_val_samples > 0 else 0

                            val_acc = metrics.accuracy_score(y_val, val_pred)
                            logger.info(f"[VAL] Accuracy: {val_acc}")
                            logger.info(f"[VAL] Average Inference Latency per repetition: {val_latency_ms:.2f} ms")

                        except Exception as val_e:
                            logger.error(f"Validation failed: {val_e}")

                    # Build the CSV row utilizing the validation latency
                    current_row = {
                        "algorithm": algorithm,
                        "exercise": exercise,
                        "seed": seed_value,
                        "train_accuracy": rocket.train_acc_,
                        "val_accuracy": val_acc,
                        "test_accuracy": rep["accuracy"],
                        "test_macro_f1": rep["macro avg"]["f1-score"],
                        "test_weighted_f1": rep["weighted avg"]["f1-score"],
                        "inference_latency_ms": val_latency_ms
                    }

                    run_rows.append(current_row)

                    # Plot CM
                    plot_confusion_matrix(exercise_out, seed_value, res["test"]["cm"], res["test"]["labels"])

                    # Save the Trained Model using Joblib
                    model_filename = os.path.join(exercise_out, f"{algorithm}_{exercise}_seed{seed_value}_model.pkl")
                    joblib.dump(rocket, model_filename)
                    logger.info(f"Saved trained model to: {model_filename}")

                except Exception as e:
                    logger.error(f"Failed on {algorithm}/{exercise}: {e}")

    # Save Summary CSV
    if run_rows:
        df = pd.DataFrame(run_rows)
        print("\n=== FINAL RESULTS SUMMARY ===")
        print(df.to_string(index=False))
        print("=============================\n")

        df.to_csv(os.path.join(output_rocket_root, "final_results_summary.csv"), index=False)
        logger.info("Saved final_results_summary.csv")
    else:
        logger.warning("No results to save.")