from fastai.vision.all import *
import pandas as pd

IMAGE_PIXEL_COUNT = 224
EPOCH_VALUE = 3

def GetLabel(fileName: str):
    return fileName.split("-")[0]

Food_database_path = untar_data(URLs.FOOD)

label1 = "ramen"
label2 = "nachos"
label3 = "pizza"
label4 = "ice_cream"
label5 = "fried_rice"
label6 = "baklava"
label7 = "dumplings"
label8 = "samosa"

for img in get_image_files(Food_database_path):

    if label1 in str(img):
        img.rename(f"{img.parent}/{label1}-{img.name}")
    
    # elif label2 in str(img):
    #     img.rename(f"{img.parent}/{label2}-{img.name}")
    
    elif label3 in str(img):
        img.rename(f"{img.parent}/{label3}-{img.name}")
    
    # elif label4 in str(img):
    #     img.rename(f"{img.parent}/{label4}-{img.name}")
    
    # elif label5 in str(img):
    #     img.rename(f"{img.parent}/{label5}-{img.name}")
    
    elif label6 in str(img):
        img.rename(f"{img.parent}/{label6}-{img.name}")
    
    # elif label7 in str(img):
    #     img.rename(f"{img.parent}/{label7}-{img.name}")
    
    # elif label8 in str(img):
    #     img.rename(f"{img.parent}/{label8}-{img.name}")
    
    else:
        os.remove(img)

# Loading images into model

Food_dls = ImageDataLoaders.from_name_func(Food_database_path,
                                           get_image_files(Food_database_path),
                                           valid_pct=0.2,
                                           see=2705,
                                           label_func=GetLabel,
                                           item_tfms=Resize(IMAGE_PIXEL_COUNT))

Food_model = cnn_learner(Food_dls ,resnet34, metrics=error_rate, pretrained=True)

Food_model.fine_tune(epochs=EPOCH_VALUE)

# Confusion Matrix

Food_model_interpretation = ClassificationInterpretation.from_learner(Food_model)
Food_model_interpretation.plot_confusion_matrix()

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

