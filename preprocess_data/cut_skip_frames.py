import cv2
import os
import glob
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN

phase = "public_test_2"
video_root = f"./original_data/videos/{phase}/"
original_gts_df = pd.read_csv(f"./original_data/groundtruths/{phase}.csv")

saved_root = f"./data/{phase}_skip_frames/"
new_gts_path = os.path.join(saved_root, f"{phase}_skip_frames.csv")
new_gts_df = pd.DataFrame(columns=["name", "label"])


if not os.path.isdir(os.path.join(saved_root, "images")):
    os.makedirs(os.path.join(saved_root, "images"))

mtcnn = MTCNN(device="cpu")


for idx, row in original_gts_df.iterrows():
    video_path = os.path.join(video_root, row["filename"])
    print("Processing", video_path)
    vid_cap = cv2.VideoCapture(video_path)

    i = -1
    SKIP_FRAMES = vid_cap.get(cv2.CAP_PROP_FPS)
    img_count = 0

    while True:
        i += 1
        res, frame = vid_cap.read()

        if not res:
            break
        if i % SKIP_FRAMES == 0:
            # face detection
            rate = 3
            small_frame = cv2.resize(
                frame, (0, 0), fx=round(1 / rate, 2), fy=round(1 / rate, 2)
            )
            norm_img = np.zeros((small_frame.shape[0], small_frame.shape[1]))
            norm_small_frame = cv2.normalize(
                small_frame, norm_img, 0, 255, cv2.NORM_MINMAX
            )
            small_rgb_frame = cv2.cvtColor(norm_small_frame, cv2.COLOR_BGR2RGB)
            # try:
            faces, probs = mtcnn.detect(small_rgb_frame)
            largest_face_area = 0
            largest_face_coords = None
            if faces is None:
                continue
            for face, prob in zip(faces, probs):
                if prob > 0.5:
                    x1, y1, x2, y2 = face
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
                    X1, Y1, X2, Y2 = (
                        int(x1 * rate),
                        int(y1 * rate),
                        int(x2 * rate),
                        int(y2 * rate),
                    )
                    area = abs(X1 - Y1) * abs(X2 - Y2)
                    if area > largest_face_area:
                        largest_face_area = area
                        largest_face_coords = (X1, Y1, X2, Y2)
            if largest_face_coords is not None:
                X1, Y1, X2, Y2 = largest_face_coords
                face_region = frame[Y1:Y2, X1:X2]
                img_path = os.path.join(
                    saved_root,
                    "images",
                    os.path.basename(video_path).replace(".mp4", "")
                    + f"-{img_count}.jpg",
                )
                img_count += 1
                # save label
                label = 0.0 if row["isFake"] else 1.0
                tmp_df = pd.DataFrame([[img_path, label]], columns=["name", "label"])
                new_gts_df = pd.concat([tmp_df, new_gts_df]).reset_index(drop=True)
                # write image
                cv2.imwrite(img_path, face_region)
                # print(img_path)

new_gts_df.to_csv(new_gts_path)
