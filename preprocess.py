import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# CSV 파일에 저장할 결과 데이터프레임을 초기화합니다.
result_data = []


# data 폴더에서 real을 포함한 이미지 파일 경로를 가져오는 함수
def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                and "real" in file.lower()
            ):
                image_paths.append(os.path.join(root, file))
    return image_paths


image_paths = get_image_paths("./data")


for image_path in image_paths:
    image = cv2.imread(image_path)

    # Mediapipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # 이미지를 RGB 형식으로 변환합니다.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 특성을 추출합니다.
    results = face_mesh.process(image_rgb)

    # 추출된 얼굴 랜드마크를 리스트로 변환합니다.
    landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

    # landmarks 리스트를 NumPy 배열로 변환합니다.
    landmarks_array = np.array(landmarks, dtype=np.float32)

    # 필요한 랜드마크 인덱스를 정의합니다.
    # 눈
    left_eye = [
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
        33,
        130,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
    ]
    right_eye = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        359,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

    # 코
    nose_tip = [
        168,
        245,
        128,
        114,
        217,
        198,
        209,
        49,
        64,
        98,
        97,
        2,
        326,
        327,
        278,
        360,
        420,
        399,
        351,
    ]

    # 입
    top_lip = [61, 185, 40, 39, 37, 0, 267, 270]
    bottom_lip = [146, 91, 181, 84, 17, 314, 405, 321]

    # 눈썹
    left_eyebrow = [336, 296, 334, 293, 276, 283, 282, 295, 285]
    right_eyebrow = [107, 66, 105, 63, 70, 53, 52, 65, 66]

    # 얼굴 윤곽
    face_contour = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
        10,
    ]

    # 눈 간의 거리와 눈 높이의 비율 계산
    left_eye_x = landmarks_array[left_eye, 0]
    left_eye_y = landmarks_array[left_eye, 1]
    right_eye_x = landmarks_array[right_eye, 0]
    right_eye_y = landmarks_array[right_eye, 1]

    eye_distance = np.linalg.norm(left_eye_x - right_eye_x)
    eye_height = np.linalg.norm(left_eye_y - right_eye_y)
    eye_ratio = eye_distance / eye_height

    # 코와 입의 거리와 얼굴 높이의 비율 계산
    nose_x = landmarks_array[2, 0]
    nose_y = landmarks_array[2, 1]
    top_lip_x = landmarks_array[top_lip, 0]
    top_lip_y = landmarks_array[top_lip, 1]
    bottom_lip_x = landmarks_array[bottom_lip, 0]
    bottom_lip_y = landmarks_array[bottom_lip, 1]

    # 코의 좌우 길이 계산
    leftmost_nose_x = landmarks_array[nose_tip[0], 0]
    rightmost_nose_x = landmarks_array[nose_tip[8], 0]
    nose_width = np.linalg.norm(leftmost_nose_x - rightmost_nose_x)

    # 코의 위아래 길이 계산
    top_nose_y = landmarks_array[nose_tip[12], 1]
    bottom_nose_y = landmarks_array[nose_tip[0], 1]
    nose_height = np.linalg.norm(top_nose_y - bottom_nose_y)

    # 얼굴 폭과 얼굴 높이의 비율 계산
    face_width = np.linalg.norm(landmarks_array[16] - landmarks_array[0])
    face_height = np.linalg.norm(landmarks_array[8] - landmarks_array[27])
    face_ratio = face_width / face_height

    # 눈썹 간의 거리와 얼굴 높이의 비율 계산
    left_eyebrow_x = landmarks_array[left_eyebrow, 0]
    left_eyebrow_y = landmarks_array[left_eyebrow, 1]
    right_eyebrow_x = landmarks_array[right_eyebrow, 0]
    right_eyebrow_y = landmarks_array[right_eyebrow, 1]

    eyebrow_distance = np.linalg.norm(left_eyebrow_x - right_eyebrow_x)
    eyebrow_height = np.linalg.norm(left_eyebrow_y - right_eyebrow_y)
    eyebrow_ratio = eyebrow_distance / eyebrow_height

    # 얼굴 윤곽의 길이 대비 얼굴 높이의 비율 계산
    face_contour_x = landmarks_array[face_contour, 0]
    face_contour_y = landmarks_array[face_contour, 1]

    face_contour_length = np.sum(
        np.linalg.norm(
            np.diff(np.column_stack((face_contour_x, face_contour_y))), axis=1
        )
    )
    face_contour_ratio = face_contour_length / face_height

    # 입술 좌우 길이 계산 (하나의 길이로 표현)
    lip_width = np.linalg.norm(
        landmarks_array[top_lip[0]] - landmarks_array[top_lip[6]]
    )

    # 입술과 코 간의 거리와 입술 높이의 비율 계산
    lip_nose_distance = np.linalg.norm(nose_x - (top_lip_x + bottom_lip_x) / 2)
    lip_nose_height = np.linalg.norm(nose_y - (top_lip_y + bottom_lip_y) / 2)
    lip_nose_ratio = lip_nose_distance / lip_nose_height

    # 입술 두께 계산 (상단과 하단의 크기 비율)
    lip_thickness = np.linalg.norm(
        landmarks_array[top_lip[0]] - landmarks_array[bottom_lip[6]]
    ) / (lip_nose_distance + 1e-6)

    # 이미지 위에 랜드마크 간의 거리를 선으로 그립니다.
    for indices in [
        left_eye,
        right_eye,
        nose_tip,
        top_lip,
        bottom_lip,
        left_eyebrow,
        right_eyebrow,
        face_contour,
    ]:
        for i in range(len(indices) - 1):
            index1 = indices[i]
            index2 = indices[i + 1]
            x1, y1, _ = landmarks_array[index1]
            x2, y2, _ = landmarks_array[index2]
            x1, y1, x2, y2 = (
                int(x1 * image.shape[1]),
                int(y1 * image.shape[0]),
                int(x2 * image.shape[1]),
                int(y2 * image.shape[0]),
            )
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    new_image_path = image_path.replace("_real", "").replace(".png", ".jpg")

    new_data = {
        "image_path": new_image_path,
        "Eye Ratio": eye_ratio,
        "Face Ratio": face_ratio,
        "Eyebrow Ratio": eyebrow_ratio,
        "Lip Width": lip_width,
        "Nose Width": nose_width,
    }

    result_data.append(new_data)

# 데이터프레임으로 변환
result_df = pd.DataFrame(result_data)

# Sort the DataFrame by 'Eye Ratio' column
result_df = result_df.sort_values(by="Eye Ratio")

# # 처리할 수치 열의 목록과 해당 별칭 정의
column_aliases = {
    "Eye Ratio": "눈",
    "Face Ratio": "얼굴",
    "Eyebrow Ratio": "눈썹",
    "Lip Width": "입",
    "Nose Width": "코",
}

# 각 수치 열에 대한 사분위수 기반으로 라벨 생성
for col, alias in column_aliases.items():
    bottom_quantile = result_df[col].quantile(0.33)
    top_quantile = result_df[col].quantile(0.67)

    # 경계선 값 출력
    print(f"Bottom Quantile ({col}): {bottom_quantile}")
    print(f"Top Quantile ({col}): {top_quantile}")

    result_df[alias + "_label"] = "중" + alias  # 기본값으로 중간 라벨 설정
    result_df.loc[result_df[col] <= bottom_quantile, alias + "_label"] = "소" + alias
    result_df.loc[result_df[col] > top_quantile, alias + "_label"] = "대" + alias

# 'caption' 열 생성: 라벨 열을 이용하여 생성
label_columns = [alias + "_label" for alias in column_aliases.values()]
result_df["caption"] = "메이플캐릭터," + result_df[label_columns].apply(
    lambda row: ",".join(row), axis=1
)

selected_columns = ["image_path", "caption"]
new_df = result_df[selected_columns]


# 수정 함수 정의
def modify_caption(row):
    if "woman" in row["image_path"]:
        return row["caption"] + ",woman"
    else:
        return row["caption"] + ",man"


# apply 함수를 사용하여 caption 열 수정
new_df["caption"] = new_df.apply(modify_caption, axis=1)


# 새로운 DataFrame을 CSV 파일로 저장
new_df.to_csv("dataset.csv", index=False)
