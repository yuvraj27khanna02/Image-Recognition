import pandas as pd


def GetLabel(fileName: str):
    return fileName.split("-")[0]

def all_food_names() -> list:
    with open("food-101/labels.txt", "r") as food_file:
        food_labels = food_file.read().splitlines()
    return food_labels

def rename_food_img(food_img: str):
    for food_name in all_food_names():
        if food_name in str(food_img):
            return str(food_name)