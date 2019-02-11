import numpy as np
import torch
import os
import json
import tkinter as tk
from tkinter.filedialog import askdirectory
from matplotlib import pyplot
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import *
from train import *
from dataset import SaliencyDataset


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def load_config(path):

    with open(path) as file:
        config = json.load(file)

    hidden_size = config["hidden size"]
    kernal_size = config["kernal size"]
    time_stamp = config["time steps"]

    return hidden_size, kernal_size, time_stamp


def load_data(x, batch_size, y=None):
    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data = SaliencyDataset(x, y, data_transforms)
    loader = DataLoader(data, batch_size=batch_size)

    return loader


def main(path):
    config_file = [file for file in os.listdir('.') if file.endswith('.json')]
    hidden_size, kernal_size, time_stamp = load_config(os.getcwd() + '/' + config_file[0])

    model_file = [file for file in os.listdir('.') if file.endswith('.pt')]
    if torch.cuda.is_available():
        print('Using GPU')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        file = torch.load(model_file[0])
    else:
        print('Using CPU')
        torch.set_default_tensor_type('torch.FloatTensor')
        file = torch.load(model_file[0], map_location='cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if type(file) == dict:
        model = CombinedModel(hidden_size, kernal_size, time_stamp)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(file['state_dict'])
    else:
        model = file
    model.to(device)
    os.chdir(path + '/images')
    file_list = [file for file in os.listdir('.') if file.endswith('.jpg')]
    data = []
    for file in file_list:
        data.append(pyplot.imread(file))
    test_loader = load_data(np.array(data), len(file_list))

    os.chdir(path)
    if not os.path.exists('maps'):
        os.makedirs('maps')
    os.chdir(path + '/maps')
    for j, data in enumerate(test_loader, 0):
        inputs = data
        outputs = model(inputs.to(device)).squeeze().cpu().detach().numpy()
        if outputs.ndim == 2:
            pyplot.imsave("map_{}.png".format(0), outputs, cmap="gray")
            continue
        for k in range(outputs.shape[0]):
            pyplot.imsave("map_{}.png".format(k), outputs[k, :, :], cmap="gray")


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

        self.master.title("Saliency Map Predictor")

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
#     path = os.getcwd()
#
#     main(path)
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()