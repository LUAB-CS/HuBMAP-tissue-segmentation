"""
Here, I make for assumption for now that the model outputs are binary segmentation arrays of the size of the
original picture.
"""

import numpy as np
import pandas as pd
def make_submission(model, test_dataset, threshold):
    prediction = model(test_dataset) # I will update the beginning of the script to fit the other's work
    submission = {'id':[], 'rle':[]}

    # From binary arrays to run-length encoding
    linestart = None
    for id, mask in enumerate(prediction):
        mask_size = mask.shape[0]
        submission["id"].append(id)
        rle = []
        binary_mask = (mask> threshold).astype(np.uint8)
        for k, line in enumerate(binary_mask):
            for l, pixel in enumerate(line):
                if pixel == 0 and linestart : # End current segment
                    lineend = k * mask_size + l - 1
                    rle = rle + f"{linestart} {lineend} "
                    linestart = None
                if linestart and l == mask_size - 1 # Segment end at border
                    lineend = k * mask_size + l
                    rle = rle + f"{linestart} {lineend} "
                    linestart = None
                elif pixel == 1 and not(linestart) : # Start new segment
                    linestart = k * mask_size

    # Create .csv
    test_df = pd.read_csv("HuBMAP-tissue-segmentation/data/" + "test.csv")
    sub = pd.DataFrame(submission)
    test_df = test_df.merge(sub, on='id')
    sub = test_df[['id', 'rle']].copy()
    sub.to_csv('submission.csv', index=False)
    # sub
