import os
import joblib

# Import the functions and class directly from your rocket.py script!
from tsc.rocket import read_dataset, RocketTransformerClassifier

if __name__ == "__main__":
    # ==========================================
    # 1. SETUP YOUR PATHS
    # ==========================================
    # Change these to point to the specific model and exercise you want to test!
    ALGORITHM = "rtmpose"
    EXERCISE = "GS"
    SEED = "103007"
    DATA_TYPE = "BodyMTS"  # Replace with whatever DATA_TYPE is in your rocket_config

    # Path to the .pkl model you want to test
    model_path = os.path.join("results", "Rocket", ALGORITHM, EXERCISE, f"{ALGORITHM}_{EXERCISE}_seed{SEED}_model.pkl")

    # Path to the data folder where the .ts files are kept
    data_path = os.path.join("data", "your_input_folder", ALGORITHM, EXERCISE,
                             SEED)  # Update "data/your_input_folder" to your actual INPUT_DATA_PATH

    # ==========================================
    # 2. LOAD THE SAVED MODEL
    # ==========================================
    print(f"Loading trained brain from: {model_path}")
    loaded_model = joblib.load(model_path)
    print("Model successfully loaded!\n")

    # ==========================================
    # 3. LOAD THE VALIDATION DATA
    # ==========================================
    print(f"Loading data from: {data_path}")
    # We use your existing read_dataset function, but we only care about x_val and y_val
    x_train, y_train, x_test, y_test, x_val, y_val, _, _ = read_dataset(data_path, DATA_TYPE)

    # ==========================================
    # 4. RUN THE PREDICTION
    # ==========================================
    if x_val is not None and len(x_val) > 0:
        print(f"\nValidation data found! Evaluating {len(x_val)} repetitions...")

        # Because your loaded_model IS the RocketTransformerClassifier class,
        # it remembers all its functions. We can just call predict_rocket!
        results = loaded_model.predict_rocket(x_val, y_val)

        print("\n=== EVALUATION COMPLETE ===")
        # Note: The logger inside predict_rocket will automatically print the
        # accuracy, inference latency, and classification report to your screen!

    else:
        print("ERROR: Could not find x_val and y_val data in that folder.")