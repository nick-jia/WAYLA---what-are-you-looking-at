import os
import sys
import random
import math
import numpy as np
import pandas as pd
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askdirectory
from scipy.io import loadmat
from utils import process_data
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd() + '/Mask_RCNN-master')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco



# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def main(curr_path):
    # image, fixation = process_data('E:\mie_project_data', 'train', 0, 1)

    os.chdir(curr_path + '/images')
    image_list = [file for file in os.listdir('.') if file.endswith('.jpg')]
    images = []
    for image in image_list:
        images.append(skimage.io.imread(image))

    os.chdir(curr_path + '/maps')
    map_list = [file for file in os.listdir('.') if file.endswith('.jpg') or file.endswith('.png')]
    maps = []
    for map in map_list:
        map_i = plt.imread(map)
        if map_i.ndim > 2:
            map_i = map_i[:, :, 0]
        maps.append(map_i)

    df = pd.DataFrame(0, index=image_list, columns=class_names)



    # Run detection
    for i in range(len(images)):

        print(i)
        results = model.detect([images[i]], verbose=0)

        # Visualize results
        r = results[0]

        score_dict = visualize.display_instances_custom(images[i], r['rois'], r['masks'], r['class_ids'],
                                                        class_names, maps[i], r['scores'])

        print(image_list[i])
        for j in score_dict.keys():
            df.loc[image_list[i], class_names[j]] = score_dict[j]
            print('{0!s} : {1:.2f} %'.format(class_names[j], score_dict[j] * 100))
    os.chdir(curr_path)
    df.to_csv('result.csv')
    print('done saving')


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.back = tk.Frame(width=150, height=50)
        self.back.grid()

        self.inputs = tk.Entry(self, bd=5, width=90)
        self.inputs.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.browser = tk.Button(self, text="Browser", command=self.get_path)
        self.browser.grid(row=1, column=0, sticky="ewns", padx=10, pady=10)

        self.auto = tk.Button(self, text="AUTO", command=self.auto)
        self.auto.grid(row=1, column=1, sticky="ewns", padx=10, pady=10)

        self.clear = tk.Button(self, text="Clear", command=self.clear)
        self.clear.grid(row=1, column=2, sticky="ewns", padx=10, pady=10)

        self.run = tk.Button(self, text="RUN", command=self.run_main)
        self.run.grid(row=1, column=3, sticky="ewns", padx=10, pady=10)

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.grid(row=2, column=0, columnspan=4, sticky="ewns", padx=10, pady=10)

        photo = tk.PhotoImage(file='wayla.gif')
        self.label = tk.Label(image=photo, width=640)
        self.label.image = photo
        self.label.grid(row=3, column=0, columnspan=4, padx=10, pady=10)

        self.master.title("What Are You Looking At")

    def run_main(self):
        main(tk.Entry.get(self.inputs))

    def auto(self):
        tk.Entry.insert(self.inputs, 0, os.getcwd())

    def get_path(self):
        path = askdirectory()
        tk.Entry.insert(self.inputs, 0, path)

    def clear(self):
        tk.Entry.delete(self.inputs, 0, 'end')


if __name__ == '__main__':
    # path = os.getcwd()
    # main(path)
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
