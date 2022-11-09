#!/usr/bin/env python3

# The main file used to execute the program, runs using pycharm.
# Calls segmentation/recogntion where required in order to
# allow program to function. Allows developers to decide between
# training and testing mode.

import os
import recognition as rec
import segmentation as seg
import sys

if len(sys.argv) != 2:
    print("Usage: %s Training? (True/False)" % sys.argv[0], file=sys.stderr)
    training = False
else:
    if str(sys.argv[1]) == 'True':
        training = True
    else:
        training = False

print("Training: "+str(training))

if training:
    seg.main_seg('../main/data/training data', training)
    print("Data has been generated, however requires sorting into classes for training.\n"
          "Please visit ../main/training/extracts_training to see the extractions made.")

    exit(1)

classes = rec.main_rec()
seg.main_seg('../main/data/images', training)

try:
    for file in os.listdir('main/training/extracts_training'):
        os.remove('main/training/extracts_training' + "/" + file)

    for file in os.listdir('main/testing/extracts_testing'):
        os.remove('main/testing/extracts_testing' + "/" + file)

except FileNotFoundError as e:
    print("\n\nFileNotFoundError; no files in given directory(s)")







