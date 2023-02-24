from fastai.vision.all import *
import matplotlib.image as mpimg
from PIL import Image

# The model is made from the code in Basic-Image-Recognition -> ImageRecognitionModel.py

def GetLabel(fileName):
    return fileName.split("-")[0]

Food_model = load_learner("FoodRecognitionModel/export.pkl")
img_test_path = "FoodRecognitionModel/images_to_test/test1.jpeg"
test_img = Image.open(img_test_path)
Food_model_pred = Food_model.predict(mpimg.imread(img_test_path))
model_categories = ["baklava", "pizza", "ramen"]

print("         ######### PREDICTION #########")
print(f" Prediction : {Food_model_pred[0]}")
for i in range(3):
    print(f"{model_categories[i]} : {Food_model_pred[2][i]}")
test_img.show()