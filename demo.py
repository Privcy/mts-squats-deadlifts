import cv2
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import traceback
import sys

import sktime.transformers.series_as_features.rocket

# Intercept modern model paths and redirect to 2021 paths
sys.modules['sktime.transformations'] = sktime.transformers
sys.modules['sktime.transformations.panel'] = sktime.transformers.series_as_features
sys.modules['sktime.transformations.panel.rocket'] = sktime.transformers.series_as_features.rocket
sys.modules['sktime.transformations.panel.rocket._rocket'] = sktime.transformers.series_as_features.rocket

from tsc.rocket import RocketTransformerClassifier
import __main__

__main__.RocketTransformerClassifier = RocketTransformerClassifier

# Import Pose Extractors
from rtmlib import BodyWithFeet


# --- EXPLAINABLE AI GRAPHING IMPORTS ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Mute warnings for a clean presentation
warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


def render_live_graph_to_image(current_frame_index, feature_24, feature_25, feature_26, feature_27):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    canvas = FigureCanvas(fig)

    frames = range(current_frame_index)
    f24_len = len(feature_24[:current_frame_index])

    if f24_len > 0:
        ax.plot(frames[:f24_len], feature_24[:current_frame_index], label='Feature 24 (Left Knee Y)', color='#1f77b4',
                linewidth=2)
        ax.plot(frames[:f24_len], feature_25[:current_frame_index], label='Feature 25 (Right Knee Y)', color='#ff7f0e',
                linewidth=2)
        ax.plot(frames[:f24_len], feature_26[:current_frame_index], label='Feature 26 (Left Hip Y)', color='#2ca02c',
                linewidth=2)
        ax.plot(frames[:f24_len], feature_27[:current_frame_index], label='Feature 27 (Right Hip Y)', color='#d62728',
                linewidth=2)

    ax.set_title("Live MTS Extraction (Raw Trajectory)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Video Frames", fontsize=12)
    ax.set_ylabel("Vertical Pixel Coordinate (Y-Axis)", fontsize=12)
    ax.invert_yaxis()

    ax.set_xlim(0, max(100, current_frame_index + 10))

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    canvas.draw()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    graph_img = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    graph_img_bgr = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return graph_img_bgr


class WorkoutFormDetector:
    def __init__(self, combined_model_path, algorithm_name):
        self.algorithm_name = algorithm_name
        logger.info(f"Loading trained model bundle: {combined_model_path}")
        self.model_bundle = joblib.load(combined_model_path)
        self.rocket = self.model_bundle.rocket

        # --- THE FINAL NUMBA PATCH ---
        if hasattr(self.rocket, 'kernels') and self.rocket.kernels is not None:
            w, l, b, d, p, n, i = self.rocket.kernels
            self.rocket.kernels = (w.astype(np.float64), l, b.astype(np.float64), d, p, n, i)

        self.classifier = self.model_bundle.classifier
        self.max_len = self.model_bundle.max_len_

        # Initialize the correct extraction engine based on terminal selection
        self.pose_estimator = self._initialize_extractor()

    def _initialize_extractor(self):
        if self.algorithm_name == "mediapipe":
            logger.info("Initializing MediaPipe Pose Engine...")
            # LAZY IMPORT: We only import MediaPipe here so it doesn't crash OpenCV at startup!
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return "mediapipe"
        else:
            # Defaults to RTMPose for both 'rtmpose' and 'openpose' logic
            logger.info("Initializing RTMPose/OpenPose Engine (rtmlib)...")
            return BodyWithFeet(to_openpose=True, device="cpu", backend="onnxruntime")

    def _extract_mediapipe_frame(self, frame):
        """Converts MediaPipe landmarks into the 27x2 array format your model expects."""
        results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        person_kps = np.zeros((27, 2))
        person_scores = np.zeros(27)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = results.pose_landmarks.landmark

            # Map MediaPipe indices to your OpenPose/RTMPose format indices
            # (This mapping must match how you extracted data in your dataset)
            mp_to_op = {
                0: 0,  # Nose
                2: 1,  # L Eye -> REye? (Mapping roughly for rendering)
                5: 2,  # R Eye -> LEye?
                11: 5,  # L Shoulder
                12: 6,  # R Shoulder
                13: 7,  # L Elbow
                14: 8,  # R Elbow
                15: 9,  # L Wrist
                16: 10,  # R Wrist
                23: 11,  # L Hip
                24: 12,  # R Hip
                25: 13,  # L Knee
                26: 14,  # R Knee
                27: 15,  # L Ankle
                28: 16  # R Ankle
            }

            for mp_idx, op_idx in mp_to_op.items():
                lm = landmarks[mp_idx]
                person_kps[op_idx] = [lm.x * w, lm.y * h]
                person_scores[op_idx] = lm.visibility

        return person_kps, person_scores

    def _calculate_angle(self, point_a, point_mid, point_b):
        a, mid, b = np.array(point_a)[:2], np.array(point_mid)[:2], np.array(point_b)[:2]
        vector_a, vector_b = a - mid, b - mid
        mag_a, mag_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)

        if mag_a == 0 or mag_b == 0: return 0.0

        dot_product = np.dot(vector_a, vector_b)
        cos_theta = np.clip(dot_product / (mag_a * mag_b), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def extract_and_normalize_keypoints(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints_over_time = []
        frame_history = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # Dynamic Extraction based on selected algorithm
            if self.algorithm_name == "mediapipe":
                person_kps, person_scores = self._extract_mediapipe_frame(frame)
            else:
                keypoints, scores = self.pose_estimator(frame)
                if keypoints is not None and len(keypoints) > 0:
                    person_kps = keypoints[0]
                    person_scores = scores[0]
                else:
                    person_kps = np.zeros((27, 2))
                    person_scores = np.zeros(27)

            person_kps_with_c = np.array(
                [[float(x), float(y), float(c)] for (x, y), c in zip(person_kps, person_scores)])

            left_leg = self._calculate_angle(person_kps[12], person_kps[13], person_kps[14])
            right_leg = self._calculate_angle(person_kps[9], person_kps[10], person_kps[11])
            left_arm = self._calculate_angle(person_kps[5], person_kps[6], person_kps[7])
            right_arm = self._calculate_angle(person_kps[2], person_kps[3], person_kps[4])
            angles = [left_leg, right_leg, left_arm, right_arm]

            frame_history.append((person_kps, angles))

            bugged_array = person_kps_with_c.copy()
            for angle in angles:
                bugged_array = np.concatenate((bugged_array, np.full((bugged_array.shape[0], 1), angle)), axis=1)

            flattened = bugged_array.flatten()
            final_79_features = np.nan_to_num(flattened[:79], nan=0.0, posinf=0.0, neginf=0.0)
            keypoints_over_time.append(final_79_features)

        cap.release()
        logger.info(f"Pass 1: Extracted physics for {frame_count} frames using {self.algorithm_name.upper()}.")
        mts_data = self._convert_to_sktime_format(keypoints_over_time)
        return mts_data, frame_history

    def _convert_to_sktime_format(self, keypoints_list):
        raw_data = np.array(keypoints_list, dtype=np.float64)
        if raw_data.shape[0] == 0: raise ValueError("No frames processed. Video may be unreadable.")

        num_frames = raw_data.shape[0]
        if num_frames > self.max_len:
            raw_data = raw_data[np.linspace(0, num_frames - 1, self.max_len).astype(int)]
            num_frames = self.max_len

        df_dict = {}
        for feature_idx in range(raw_data.shape[1]):
            feature_series = raw_data[:, feature_idx]
            if num_frames < self.max_len:
                pad_val = feature_series[-1]
                feature_series = np.concatenate(
                    [feature_series, np.full(self.max_len - num_frames, pad_val, dtype=np.float64)])
            df_dict[f"feature_{feature_idx}"] = [pd.Series(feature_series, dtype=np.float64)]

        return pd.DataFrame(df_dict)

    def draw_skeleton_and_angles(self, frame, keypoints, angles):
        coco_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]

        for pair in coco_pairs:
            partA, partB = pair[0], pair[1]
            if partA < len(keypoints) and partB < len(keypoints):
                pt1 = (int(keypoints[partA][0]), int(keypoints[partA][1]))
                pt2 = (int(keypoints[partB][0]), int(keypoints[partB][1]))
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(frame, pt1, pt2, (255, 150, 0), 3)
                    cv2.circle(frame, pt1, 5, (0, 0, 255), -1)
                    cv2.circle(frame, pt2, 5, (0, 0, 255), -1)

        joint_indices = [13, 14, 7, 8]
        for angle, joint_idx in zip(angles, joint_indices):
            if joint_idx < len(keypoints):
                pt = (int(keypoints[joint_idx][0]), int(keypoints[joint_idx][1]))
                if pt != (0, 0):
                    text = f"{int(angle)}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (pt[0] + 10, pt[1] - 25),
                                  (pt[0] + 10 + text_width, pt[1] - 25 + text_height + 5), (0, 0, 0), -1)
                    cv2.putText(frame, text, (pt[0] + 10, pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def predict_and_annotate(self, video_path, output_video_path):
        logger.info(f"Analyzing repetition dynamics...")

        # --- PASS 1: EXTRACT & PREDICT ---
        mts_data, frame_history = self.extract_and_normalize_keypoints(video_path)

        logger.info("Classifying execution via ROCKET engine...")
        rocket_features = self.rocket.transform(mts_data)
        rocket_features = np.clip(np.nan_to_num(rocket_features, nan=0.0, posinf=1e10, neginf=-1e10), -1e10, 1e10)

        prediction = self.classifier.predict(rocket_features)[0]
        is_correct = (prediction == "correct")

        status_text = "CORRECT EXECUTION" if is_correct else "INCORRECT (Form Error)"
        color = (0, 255, 0) if is_correct else (0, 0, 255)

        # Pre-extract Y trajectories for XAI graph
        raw_y_24 = [kps[13][1] for kps, ang in frame_history]  # Left Knee
        raw_y_25 = [kps[14][1] for kps, ang in frame_history]  # Right Knee
        raw_y_26 = [kps[11][1] for kps, ang in frame_history]  # Left Hip
        raw_y_27 = [kps[12][1] for kps, ang in frame_history]  # Right Hip

        # --- PASS 2: RENDER DASHBOARD & EXPORT ---
        logger.info("Rendering live XAI dashboard...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, first_frame = cap.read()
        if not ret: return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        test_graph = render_live_graph_to_image(1, raw_y_24, raw_y_25, raw_y_26, raw_y_27)
        test_graph_resized = cv2.resize(test_graph, (test_graph.shape[1], first_frame.shape[0]))
        test_dashboard = cv2.hconcat([first_frame, test_graph_resized])
        h, w, _ = test_dashboard.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        window_name = "Live Inference Dashboard (XAI View)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_idx < len(frame_history):
                keypoints, angles = frame_history[frame_idx]

                self.draw_skeleton_and_angles(frame, keypoints, angles)

                cv2.rectangle(frame, (10, 10), (600, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Analysis: {status_text}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # Update UI Text to reflect the active model
                model_display_name = "MediaPipe" if self.algorithm_name == "mediapipe" else (
                    "OpenPose" if self.algorithm_name == "openpose" else "RTMPose")
                cv2.putText(frame, f"Model: {model_display_name} + ROCKET Engine", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            graph_frame = render_live_graph_to_image(frame_idx + 1, raw_y_24, raw_y_25, raw_y_26, raw_y_27)
            graph_frame_resized = cv2.resize(graph_frame, (graph_frame.shape[1], frame.shape[0]))
            combined_dashboard = cv2.hconcat([frame, graph_frame_resized])

            out.write(combined_dashboard)
            cv2.imshow(window_name, combined_dashboard)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1

        # ========================================================
        # NEW: EXPORT HIGH-RESOLUTION STANDALONE GRAPH (300 DPI)
        # ========================================================
        # (Notice how this is now safely OUTSIDE the while loop!)
        base_vid_name = os.path.basename(output_video_path)
        dir_name = os.path.dirname(output_video_path)
        graph_filename = base_vid_name.replace(".mp4", "_HighRes_Graph.png")
        output_graph_path = os.path.join(dir_name, graph_filename)

        # Mathematically redraw a pristine, publication-quality figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        frames = range(frame_idx)

        if len(raw_y_24[:frame_idx]) > 0:
            ax.plot(frames, raw_y_24[:frame_idx], label='Feature 24 (Left Knee Y)', color='#1f77b4', linewidth=2.5)
            ax.plot(frames, raw_y_25[:frame_idx], label='Feature 25 (Right Knee Y)', color='#ff7f0e', linewidth=2.5)
            ax.plot(frames, raw_y_26[:frame_idx], label='Feature 26 (Left Hip Y)', color='#2ca02c', linewidth=2.5)
            ax.plot(frames, raw_y_27[:frame_idx], label='Feature 27 (Right Hip Y)', color='#d62728', linewidth=2.5)

        ax.set_title(f"MTS Extraction Trajectory - {status_text}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Video Frames", fontsize=14)
        ax.set_ylabel("Vertical Pixel Coordinate (Y-Axis)", fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1, frame_idx))
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_graph_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"[SUCCESS] High-Res publication-quality graph saved to: {output_graph_path}")

        # --- Pause on the final frame ---
        logger.info("Video finished. Press any key in the video window to close...")
        if 'combined_dashboard' in locals():
            cv2.putText(combined_dashboard, "FINISHED - PRESS ANY KEY TO CLOSE", (20, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow(window_name, combined_dashboard)
            cv2.waitKey(0)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"[SUCCESS] Exported Dashboard video saved to: {output_video_path}")
        return status_text

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BodyMTS - Stiff Leg Deadlift / Squats Classifier ")
    print("=" * 60)

    print("\nAvailable Pose Estimation Frameworks:")
    print("  [1] RTMPose (Recommended/State-of-the-Art)")
    print("  [2] MediaPipe")
    print("  [3] OpenPose")
    model_choice = input("Select Model (1/2/3) [Default: 1]: ").strip()

    model_map = {"1": "rtmpose", "2": "mediapipe", "3": "openpose"}
    ALGORITHM = model_map.get(model_choice, "rtmpose")

    print("\nAvailable Exercise Variations:")
    print("  [1] Back Squat (BS)")
    print("  [2] Dumbbell Squat (DS)")
    print("  [3] Front Squat (FS)")
    print("  [4] Goblet Squat (GS)")
    print("  [5] Stiff-Leg Deadlift (SD)")
    ex_choice = input("Select Exercise (1-5) [Default: 1]: ").strip()
    ex_map = {"1": "BS", "2": "DS", "3": "FS", "4": "GS", "5": "SD"}
    EXERCISE = ex_map.get(ex_choice, "BS")

    print("\nPaste the video file path.")
    print("(You can right-click the video file in Windows and select 'Copy as path')")
    raw_video_path = input("Video Path: ").strip()

    TEST_VIDEO_PATH = raw_video_path.strip('"').strip("'")

    BASE_PATH = os.getcwd()
    MODEL_PATH = os.path.join(BASE_PATH, "figs", "Rocket", ALGORITHM, EXERCISE,
                              f"{ALGORITHM}_{EXERCISE}_seed103007_model.pkl")

    dir_name = os.path.dirname(TEST_VIDEO_PATH) if TEST_VIDEO_PATH else BASE_PATH
    base_name = os.path.basename(TEST_VIDEO_PATH) if TEST_VIDEO_PATH else "test"
    name, ext = os.path.splitext(base_name)
    OUTPUT_VIDEO_PATH = os.path.join(dir_name, f"{name}_{ALGORITHM}_XAI_Export.mp4")

    print("\n" + "-" * 60)
    print(f"LOADING CONFIGURATION:")
    print(f"-> Algorithm : {ALGORITHM.upper()}")
    print(f"-> Exercise  : {EXERCISE}")
    print(f"-> Model File: {MODEL_PATH}")
    print(f"-> Video File: {TEST_VIDEO_PATH}")
    print("-" * 60 + "\n")

    try:
        if not os.path.exists(TEST_VIDEO_PATH) and TEST_VIDEO_PATH != "":
            logger.error(f"Cannot find video at: {TEST_VIDEO_PATH}. Please check your path.")
        else:
            # Pass the algorithm name down to the detector
            demo = WorkoutFormDetector(combined_model_path=MODEL_PATH, algorithm_name=ALGORITHM)
            demo.predict_and_annotate(TEST_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
        traceback.print_exc()