import torch
from bot.environment import Board, Environment, Card
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2

decode_rank = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}

decode_suit = {"d": 0, "h": 1, "s": 2, "c": 3}

class_to_idx = {}
for rank in decode_rank.keys():
    for suit in decode_suit.keys():
        class_to_idx[rank+suit] = decode_suit[suit] * 13 + decode_rank[rank]

def image_to_card_df(model: YOLO, image_path: str, size: int = 640, unique: bool=False) -> pd.DataFrame:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    results = model(image)
    class_converter = ['Tc', 'Td', 'Th', 'Ts', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs']

    data = pd.DataFrame(results[0].boxes.data.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2', 'Confidence', 'Class'])
    data['Class'] = data['Class'].astype(int)
    data['Class'] = data['Class'].apply(class_converter.__getitem__)
    groups_mean = data.groupby("Class").mean()
    groups_size = data.groupby("Class").size()
    groups_size.name = "Size"
    grouped = pd.merge(groups_mean, groups_size, left_index=True, right_index=True)
    if unique:
        grouped = grouped[grouped['Confidence'] > 0.6]
    else:
        grouped = grouped[(grouped['Size'] >= 2) & (grouped['Confidence'] > 0.6)]
    grouped['x'] = (grouped['x1'] + grouped['x2']) / 2
    grouped['y'] = (grouped['y1'] + grouped['y2']) / 2
    grouped.drop(["x1", "x2", "y1", "y2"], axis=1, inplace=True)
    grouped.reset_index(inplace=True)

    return grouped


def image_to_board(model: YOLO, image_path: str, size: int = 640, max_gap:int = 20) -> Board:
    grouped = image_to_card_df(model, image_path, size, unique=False)

    streets = [[], [], []]

    grouped.sort_values("y", ascending=False, inplace=True, ignore_index=True)

    last = 2
    ci = Card.new(grouped.loc[0, "Class"])
    streets[last].append(ci)
    for i in range(1, grouped.shape[0]):
        ci = Card.new(grouped.loc[i, "Class"])
        if grouped.loc[i-1, 'y'] - grouped.loc[i, "y"] > max_gap:
            last -= 1
        streets[last].append(ci)
    return Board(streets[0], streets[1], streets[2])



def image_to_hand(model: YOLO, image_path: str, size: int = 640, max_gap:int = 20) -> list[int]:
    grouped = image_to_card_df(model, image_path, size, unique=True)
    hand = []
    for i in range(grouped.shape[0]):
        ci = Card.new(grouped.loc[i, "Class"])
        hand.append(ci)
    
    return hand


            


    