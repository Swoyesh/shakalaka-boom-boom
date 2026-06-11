import cv2

# import io
# import cv2
# import numpy as np
# import requests
# from PIL import Image
# def generate_image(canvas, label):
#     prompt = label.replace(" ", "%20")
#     url = f"https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true"
#     response = requests.get(url, timeout=60)
#     print(response.status_code, response.content[:200])
#     if response.status_code != 200:
#         print(f"Generation failed: {response.status_code}")
#         return None
#     img = Image.open(io.BytesIO(response.content))
#     return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# if __name__ == "__main__":
#     result = generate_image(None, "pizza")
#     cv2.imshow("Generated", result)
#     cv2.waitKey(0)
## ControlNet Scribble setup
#
from gradio_client import Client, handle_file
from PIL import Image


def generate_image(canvas, label):
    inverted_canvas = cv2.bitwise_not(canvas)
    pil_image = Image.fromarray(cv2.cvtColor(inverted_canvas, cv2.COLOR_BGR2RGB))
    temp_path = "temp_control.png"
    pil_image.save(temp_path)
    client = Client("https://6f33284773dd143f54.gradio.live")
    result = client.predict(handle_file(temp_path), label, api_name="/predict")
    img = cv2.imread(result)
    return img
