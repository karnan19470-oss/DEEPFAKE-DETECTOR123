import sys
from elite_predictor import load_model, predict_image  # type: ignore

MODEL_PATH = "elite_resnet_detector.pth"

model = load_model(MODEL_PATH)

if len(sys.argv) < 2:

    print("Usage: python predict_image.py image.jpg")
    exit()

image_path = sys.argv[1]

result = predict_image(model, image_path)

print("\nRESULT")
print(result)