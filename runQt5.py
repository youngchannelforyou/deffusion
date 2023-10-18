import sys
import cv2
import mediapipe as mp
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSizePolicy,
)
from PyQt5.QtWidgets import QPushButton, QTextEdit, QVBoxLayout

from PyQt5.QtWidgets import QSplitter
from PyQt5.QtGui import QPixmap, QImage, QColor

import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


class FaceLandmarkApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Landmark Detection")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        # QSplitter를 사용하여 좌우로 2분할
        self.splitter = QSplitter()
        self.layout.addWidget(self.splitter)

        # 좌측 영상 출력용 레이블
        self.video_label = QLabel()
        self.splitter.addWidget(self.video_label)
        self.video_label.setAlignment(Qt.AlignCenter)  # 중앙 정렬

        # 우측 빈 위젯
        self.right_widget = QWidget()
        self.splitter.addWidget(self.right_widget)

        # 우측 위젯의 레이아웃을 QVBoxLayout으로 설정
        right_layout = QVBoxLayout(self.right_widget)

        # 버튼 추가
        self.capture_button = QPushButton("Capture")
        right_layout.addWidget(self.capture_button)

        # 텍스트 박스 추가 (여기서는 QTextEdit 사용)
        self.text_box = QTextEdit()
        right_layout.addWidget(self.text_box)

        # 스트레치 추가 (비율 1)
        right_layout.addStretch(1)

        # 두 번째 버튼 추가
        self.generate_button = QPushButton("Generate")
        right_layout.addWidget(self.generate_button)

        # 스트레치 추가 (비율 2)
        right_layout.addStretch(2)

        # 이미지 영역 추가 (우측 하단)
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 회색 박스 이미지 생성
        gray_image = QImage(640, 480, QImage.Format_RGB888)
        gray_image.fill(QColor(192, 192, 192))  # 회색으로 채우기

        # QLabel에 회색 박스 이미지 설정
        pixmap = QPixmap.fromImage(gray_image)
        self.image_label.setPixmap(pixmap)

        right_layout.addWidget(self.image_label)

        # 스트레치 추가 (비율 6)
        right_layout.addStretch(6)

        # 이미지 레이블과 버튼들의 크기를 비율에 맞게 조절
        self.capture_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.text_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 버튼 클릭 시 동작 설정
        self.capture_button.clicked.connect(self.capture)
        self.generate_button.clicked.connect(self.generate)

        # 좌측 영상과 우측 위젯의 크기 비율 조절
        self.splitter.setSizes([self.width() / 2, self.width() / 2])

        # 웹캠 프레임 업데이트를 처리할 함수
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # 웹캠 프레임 업데이트 속도 (10ms마다)

        # Mediapipe 초기화
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)

    def generate(self):
        # 텍스트 박스의 텍스트를 읽어서 변수에 저장
        prompt = self.text_box.toPlainText().strip()

        # 텍스트 박스 초기화
        self.text_box.clear()

        # Flask 웹 애플리케이션의 URL 설정
        app_url = "http://127.0.0.1:5556"  # 웹 애플리케이션의 URL을 적절히 수정해야 합니다.

        data = {"prompt": prompt}
        response = requests.get(f"{app_url}/generate", data=data)

        if response.status_code == 200:
            # 이미지 바이트 데이터를 QImage로 변환
            image_data = response.content
            q_image = QImage.fromData(image_data)

            # QImage를 QPixmap으로 변환하여 이미지 라벨에 표시
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def capture(self):
        # 웹캠에서 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            return

        # 좌우반전
        frame = cv2.flip(frame, 1)

        # 이미지 크기 얻기
        image_height, image_width, _ = frame.shape

        # 이미지를 RGB 형식으로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 특성 추출
        results = self.face_mesh.process(image_rgb)

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

        # 입술 좌우 길이 계산 (하나의 길이로 표현)
        lip_width = np.linalg.norm(
            landmarks_array[top_lip[0]] - landmarks_array[top_lip[6]]
        )

        caption = "메이플캐릭터,"

        if eye_ratio < 6.096545581817628:
            caption += "대눈,"
        elif eye_ratio > 9.090017929077149:
            caption += "소눈,"
        else:
            caption += "중눈,"

        if face_ratio < 0.32334879755973817:
            caption += "대얼굴,"
        elif face_ratio > 0.3679878675937653:
            caption += "소얼굴,"
        else:
            caption += "중얼굴,"

        if eyebrow_ratio < 17.789510231018067:
            caption += "대눈썹,"
        elif eyebrow_ratio > 21.317358169555668:
            caption += "소눈썹,"
        else:
            caption += "중눈썹,"

        if lip_width < 0.04647548146545887:
            caption += "대입,"
        elif lip_width > 0.0533718939870596:
            caption += "소입,"
        else:
            caption += "중입,"

        if nose_width < 0.021084710955619812:
            caption += "대코,"
        elif nose_width > 0.025440793633461:
            caption += "소코,"
        else:
            caption += "중코,"

        # 캡션을 텍스트 박스에 표시
        self.text_box.setPlainText(caption)

    def update_frame(self):
        # 웹캠에서 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            return

        # 좌우반전
        frame = cv2.flip(frame, 1)

        # 이미지 크기 얻기
        image_height, image_width, _ = frame.shape

        # 이미지를 RGB 형식으로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 특성 추출
        results = self.face_mesh.process(image_rgb)

        # 추출된 얼굴 랜드마크를 리스트로 변환
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # 각 요소별 인덱스 정의
                    left_eye_indices = [
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
                    right_eye_indices = [
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
                    nose_indices = [
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
                    top_lip_indices = [61, 185, 40, 39, 37, 0, 267, 270]
                    bottom_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321]
                    left_eyebrow_indices = [336, 296, 334, 293, 276, 283, 282, 295, 285]
                    right_eyebrow_indices = [107, 66, 105, 63, 70, 53, 52, 65, 66]
                    face_contour_indices = [
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

                    # 각 요소별 인덱스에 해당하는 랜드마크만 그리기
                    if idx in left_eye_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in right_eye_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in nose_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in top_lip_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in bottom_lip_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in left_eyebrow_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in right_eyebrow_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    elif idx in face_contour_indices:
                        x, y, _ = (
                            landmark.x * image_width,
                            landmark.y * image_height,
                            landmark.z,
                        )
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 이미지를 화면에 표시 (비율 유지)
        h, w, c = frame.shape
        max_width = self.video_label.width()
        max_height = self.video_label.height()
        ratio = min(max_width / w, max_height / h)
        new_width = int(w * ratio)
        new_height = int(h * ratio)
        frame = cv2.resize(frame, (new_width, new_height))
        bytes_per_line = 3 * new_width
        q_image = QImage(
            frame.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        # 어플리케이션 종료 시 웹캠 해제
        self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FaceLandmarkApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
