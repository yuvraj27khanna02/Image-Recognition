from fastai.vision.all import *
import pandas as pd
from ImageRecognitionModelDefinitions import *
import pickle

VERBOSE = True
IMAGE_PIXEL_COUNT = 12
EPOCH_VALUE = 2

Food_database_path = untar_data(URLs.FOOD)
if VERBOSE == True:
    print("Database imported 1/6")
    print(f"            Length of database is {len(get_image_files(Food_database_path))}")

for food_img in get_image_files(Food_database_path):
    if (any(food_name in str(food_img) for food_name in all_food_names())):
        # valid food_name is in food_img
        food_img.rename(rename_food_img(str(food_img)))
    else:
        # food_img is not of food
        os.remove(food_img)
        
if VERBOSE == True:
    print("Images named changd 2/6")

# Loading images into model

Food_dls = ImageDataLoaders.from_name_func(Food_database_path,
                                           get_image_files(Food_database_path),
                                           valid_pct=0.2,
                                           see=2705,
                                           label_func=GetLabel,
                                           item_tfms=Resize(IMAGE_PIXEL_COUNT))
if VERBOSE == True:
    print("Images loaded 3/6")

Food_model = cnn_learner(Food_dls ,resnet34, metrics=error_rate, pretrained=True)
if VERBOSE == True:
    print("Model trained 4/6")

Food_model.fine_tune(epochs=EPOCH_VALUE)
if VERBOSE == True:
    print("Model fine tuned 5/6")

# Confusion Matrix

Food_model_interpretation = ClassificationInterpretation.from_learner(Food_model)
Food_model_interpretation.plot_confusion_matrix()
print(Food_model_interpretation) 

# Deploying model

export_model_input = input(str("do you want to export this model?"))
if export_model_input.lower() == "y":
    Food_model.export()
    # To store model
    food_Model_Path = get_files(Food_database_path, ".pkl")[0]
    # To load model
    food_model_info = load_learner(food_Model_Path)
    print(f"Food model is stored as: {food_Model_Path}")
    print(f"Food model info: {food_model_info}")

# Downloading model

shutil.move("food-101/export(v2).pkl", "./")
if VERBOSE == True:
    print("model exported 6/6")