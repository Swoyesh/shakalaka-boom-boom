# Shakalaka Boom Boom — Magic Pencil ✏️

Inspired by the Indian children's show *Shakalaka Boom Boom*, this project lets you draw in the air using your hand as a magic pencil. Hold a pencil-grip pose in front of your webcam, draw a sketch, show a thumbs up — and the app recognizes what you drew and generates a real image of it.

![Shakalaka Logo](assets/logo_shakalaka.png)

---

## How it works

1. **Launch** the app and pick your drawing hand (R or L)
2. **Hold** a pencil-grip pose (index + middle fingers pinched) to activate the magic pencil
3. **Draw** anything in the air — strokes appear on screen in real time
4. **Show a thumbs up** with your other hand to trigger sketch recognition
5. The model returns its **top 3 predictions** — press 1, 2, or 3 to confirm

---

## Features

- Real-time hand tracking via **MediaPipe** (21 landmarks)
- Pencil and eraser modes with sound feedback
- Sketch recognition using **EfficientNet-B0** fine-tuned on Google QuickDraw data
- 15 drawable categories: apple, cat, crown, donut, fish, flower, house, moon, pizza, smiley face, snowman, star, sun, tree, umbrella
- AI image generation from confirmed sketches
- Colorful kid-friendly UI with the Shakalaka theme

---

## Tech Stack

| Component | Tool |
|---|---|
| Hand detection | MediaPipe Hands |
| Video & canvas | OpenCV |
| Sketch recognition | EfficientNet-B0 (PyTorch) |
| Training data | Google QuickDraw dataset |
| Image generation | pollinations.ai REST API |
| Sound | pygame |

---

## Setup

```bash
git clone https://github.com/Swoyesh/shakalaka-boom-boom.git
cd shakalaka-boom-boom
pip install -r requirements.txt
python main.py
```

Requires Python 3.10+ and a webcam.

---

## Controls

| Key | Action |
|---|---|
| `E` | Toggle eraser mode |
| `C` | Clear canvas |
| `1` / `2` / `3` | Confirm prediction |
| `Q` | Quit |

---

## Project Structure

```
main.py              # Main app loop
train.py             # Model definitions (EfficientNet + QuickDNN)
download_data.py     # QuickDraw dataset downloader
utils/
  canvas.py          # Stroke drawing logic
  gesture.py         # Pencil grip + thumbs up detection
  overlay.py         # Pencil/eraser image overlay
  preprocess.py      # Canvas preprocessing for inference
  generate.py        # AI image generation
assets/              # Images, sounds, MediaPipe model
models/              # Trained model weights + category list
```
