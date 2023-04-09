"""
Here, I make for assumption for now that the model outputs are binary segmentation arrays of the size of the
original picture.
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms


def make_submission(model, device, test_df, test_dataset, threshold):
    submission = {'id':[], 'rle':[]}
    linestart = None
    for id, image, organ in test_dataset:
        mask = model(torch.unsqueeze(image, dim=0).to(device)).cpu().detach().numpy() # Depend on the test dataloader
        submission["id"].append(id)
        rle = ""
        binary_mask = (mask> threshold).astype(np.uint8)[0][0]
        mask_size = binary_mask.shape[0]
        for k, line in enumerate(binary_mask):
            for l, pixel in enumerate(line):
                if pixel == 0 and linestart : # End current segment
                    lineend = k * mask_size + l - 1
                    rle = rle + f"{linestart} {lineend} "
                    linestart = None
                if linestart and l == mask_size - 1:# Segment end at border
                    lineend = k * mask_size + l
                    rle = rle + f"{linestart} {lineend} "
                    linestart = None
                elif pixel == 1 and not(linestart) : # Start new segment
                    linestart = k * mask_size
                    print(linestart)
        submission['rle'].append(rle)

    # Create .csv
    #test_df = pd.read_csv("HuBMAP-tissue-segmentation/data/" + "test.csv")
    print(submission)
    sub = pd.DataFrame(submission)
    sub.to_csv('submission.csv', index=False)

def rle_encode(img):
    """ TBD

    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    img = img.T
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_submission2(model, device, test_df, test_dataset, threshold):
    submission = {'id':[], 'rle':[]}
    for idx, image, organ in test_dataset:
        mask = model(torch.unsqueeze(image, dim=0).to(device)) # Depend on the test dataloader
        img_row =  test_df[test_df['id']==int(idx)]
        height, width = img_row['img_height'].item(), img_row['img_width'].item()
        resized_mask = transforms.Resize((height, width))(mask).cpu().detach().numpy()
        submission["id"].append(idx)
        binary_mask = (resized_mask> threshold).astype(np.uint8)[0][0]
        submission['rle'].append(rle_encode(binary_mask))
    # Create .csv
    #test_df = pd.read_csv("HuBMAP-tissue-segmentation/data/" + "test.csv")
    sub = pd.DataFrame(submission)
    sub.to_csv('submission.csv', index=False)