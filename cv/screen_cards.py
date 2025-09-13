import mss
import numpy as np
import torch
from cv.cnn import CNN, rank_converter, suit_converter
import cv2


def take_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        return img


def preprocess_cards_from_screen(screen_np, coords):
    tensors = []
    for x1, y1, x2, y2 in coords:
        card_img = screen_np[y1:y2, x1:x2]

        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (50, 70), interpolation=cv2.INTER_AREA)

        tensor = torch.tensor(resized / 255, dtype=torch.float32).unsqueeze(0)

        tensors.append(tensor)

    return torch.stack(tensors)


def decode_predictions(preds):
    return ["" if p == 52 else rank_converter[p % 13] + suit_converter[p // 13] for p in preds]


def get_streets_card_coords(w, h, n_player=1):
    if n_player == 1:
        left_rel = 0.211
        right_rel = 0.3695
        top_rel = 0.385
        bottom_rel = 0.605
    elif n_player == 0:
        left_rel = 0.405
        right_rel = 0.595
        top_rel = 0.53
        bottom_rel = 0.81


    left = int(w * left_rel)
    right = int(w * right_rel)
    top = int(h * top_rel)
    bottom = int(h * bottom_rel)


    total_width = right - left
    total_height = bottom - top

    card_width = total_width // 5
    card_height = total_height // 3

    coords = []
    for row in range(3):
        for col in range(5):
            x1 = left + col * card_width + 2
            y1 = top + row * card_height + 2
            x2 = x1 + card_width - 3
            y2 = y1 + card_height - 3
            coords.append((x1, y1, x2, y2))

    return coords[1:4] + coords[5:]


def get_hand_card_coords(w, h, n_move=1):
    left_rel = 0.424
    right_rel = 0.578
    top_rel = 0.405
    bottom_rel = 0.475

    n_cards = 5

    left = int(w * left_rel)
    right = int(w * right_rel)
    top = int(h * top_rel)
    bottom = int(h * bottom_rel)

    total_width = right - left
    card_width = total_width // n_cards
    card_height = bottom - top

    coords = []
    for col in range(n_cards):
        x1 = left + col * card_width + 1
        y1 = top
        x2 = x1 + card_width - 1
        y2 = y1 + card_height
        coords.append((x1, y1, x2, y2))

    return coords


def get_fantasy_card_coords(w, h):
    left_rel = 0.39
    right_rel = 0.613
    top_rel = 0.245
    bottom_rel = 0.4


    left = int(w * left_rel)
    right = int(w * right_rel)
    top = int(h * top_rel)
    bottom = int(h * bottom_rel)


    total_width = right - left
    total_height = bottom - top

    card_width = total_width // 7
    card_height = total_height // 2

    coords = []
    for row in range(2):
        for col in range(7):
            x1 = left + col * card_width + 3
            y1 = top + row * card_height + 3
            x2 = x1 + card_width - 7
            y2 = y1 + card_height - 7
            coords.append((x1, y1, x2, y2))

    return coords


def predict_cards(screen, coords, model):
    input_batch = preprocess_cards_from_screen(screen, coords)
    output = model(input_batch)
    preds = output.argmax(dim=1).tolist()
    cards = decode_predictions(preds)
    return cards