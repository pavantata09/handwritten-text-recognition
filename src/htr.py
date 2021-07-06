import argparse
import cv2
import h5py
import os
import string
import datetime

from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset
from network.model import HTRModel

if __name__ == "__main__":
    
    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]
    target_path='/data2/pavan/handwritten-text-recognition/output/iam/flor/checkpoint_weights.hdf5'
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)
    image=input('enter your image path:')
    img = pp.preprocess(image, input_size=input_size)
    x_test = pp.normalization([img])

    model = HTRModel(architecture='flor',
                     input_size=input_size,
                     vocab_size=tokenizer.vocab_size,
                     top_paths=10)

    model.compile()
    model.load_checkpoint(target=target_path)

    predicts, probabilities = model.predict(x_test, ctc_decode=True)
    predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

    print("\n####################################")
    for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
        print("\nProb.  - Predict")

        for (pd, pb) in zip(pred, prob):
            print(f"{pb:.4f} - {pd}")

        cv2.imshow(f"Image {i + 1}", cv2.imread(image))
    print("\n####################################")
    cv2.waitKey(0)
