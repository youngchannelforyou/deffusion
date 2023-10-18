from flask import Flask, render_template, request, jsonify, send_file
import keras_cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import io

app = Flask(__name__)

# 미리 학습된 모델 로드
weights_path = "./frist_diffusion.h5"
img_height = img_width = 512
pokemon_model = keras_cv.models.StableDiffusion(
    img_width=img_width, img_height=img_height
)
pokemon_model.diffusion_model.load_weights(weights_path)


@app.route("/generate", methods=["GET"])
def index():
    if request.method == "GET":
        # POST 요청에서 프롬프트 데이터 가져오기
        prompt = request.form["prompt"]
        images_to_generate = 1

        # 텍스트를 이미지로 변환
        generated_images = pokemon_model.text_to_image(
            prompt, batch_size=images_to_generate, unconditional_guidance_scale=40
        )

        # 이미지를 바이트 스트림으로 변환
        image_byte_array = io.BytesIO()
        # 이미지 크기 설정
        image_height, image_width = generated_images[0].shape[
            :2
        ]  # 첫 번째 이미지의 크기를 기준으로 설정
        image_byte_array = io.BytesIO()

        # 이미지 크기를 800x600으로 고정하고 저장
        fixed_size_image = cv2.resize(generated_images[0], (800, 600))
        plt.imsave(image_byte_array, fixed_size_image.astype(np.uint8), format="png")
        image_byte_array.seek(0)
        image_byte_array.seek(0)

        # 이미지를 브라우저에게 반환
        return send_file(image_byte_array, mimetype="image/png")


if __name__ == "__main__":
    app.run(port=5556)
