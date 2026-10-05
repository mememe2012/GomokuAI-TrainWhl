import numpy as np
from rich import console
from . import Convolution as Conv
import zipfile
import json
import hashlib
import io
import os
from .ActFunc import *

CONSOLE = console.Console()

def _xavier_uniform(
    fan_in: int, fan_out: int, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Create Xavier-uniform weights for a layer's input and output widths."""
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(
        -limit, limit, size=shape if shape is not None else (fan_in, fan_out)
    )

class InitModel:
    def __init__(self):
        self.model: dict[str, np.ndarray] = {}

    def CreateModel(self, layers:list):
        """
        Input Layer : 2 * 15 * 15
        - 0 = empty
        - 1 = chess
          - the 1st board = black
          - the 2nd board = white

        Hidden Layer : layers
        - index = which layer
        - lst[index] = number of neurons
        Output Layer : 1
        - sigmoid
        """

        self.model = {}

        CONSOLE.print("[INFO]Initializing model", style="bold")

        # init convolution
        CONSOLE.print("[INFO]Initializing convolution", style="bold")
        self.model["conv1"] = _xavier_uniform(2 * 3 * 3, 8 * 3 * 3, (4, 3, 3))
        self.model["conv2"] = _xavier_uniform(8 * 3 * 3, 64 * 3 * 3, (8, 3, 3))
        CONSOLE.print("[Success]Initialized convolution", style="bold green")

        # init network
        CONSOLE.print("[INFO]Initializing network", style="bold")
        self.model["w0"] = _xavier_uniform(1024, layers[0])
        self.model["b0"] = np.zeros((1, layers[0]))

        for i, layer in enumerate(layers):
            next_layer = layers[i + 1] if i + 1 < len(layers) else 1
            self.model[f"w{i+1}"] = _xavier_uniform(layer, next_layer)
            self.model[f"b{i+1}"] = np.zeros((1, next_layer))

            CONSOLE.print(f" -Initialized layer {i+1} with {layer} neurons", style="bold")

        CONSOLE.print("[Success]Model initialized", style="bold green")

    def SaveModel(self, path:str, name:str, others:dict):
        """
        Save model to file
        - others: dict of other information to save
        """
        CONSOLE.print("[INFO]Saving model...", style="bold")
        info = {}

        # create a zip file
        with zipfile.ZipFile(f"{path}/{name}.gmdl", "w") as zipf:
          # add the model to the zip file
          for layer, value in self.model.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            buffer.seek(0)
            zipf.writestr(f"{layer}.npy", buffer.read())
            CONSOLE.print(f" -Saved layer {layer} to file", style="bold")

          # get all npy files hash
          for file in zipf.namelist():
              if file.endswith(".npy"):
                  hash = hashlib.sha1(zipf.read(file)).hexdigest()
                  info[file] = hash

          info["others"] = others

          # add json file with the info of the model
          zipf.writestr("info.json", json.dumps(info))

          CONSOLE.print(f"[Success]Saved model to file {name}.gmdl, file size = {os.path.getsize(f'{path}/{name}.gmdl') / 1024 / 1024:.2f} MiB", style="bold green")

    def LoadModel(self, path:str):
       """
       Load model from file
       """
       self.model = {}
       # load the info.json file
       CONSOLE.print("[INFO]Loading metadata file...", style="bold")
       with zipfile.ZipFile(path, "r") as zipf:
          if "info.json" not in zipf.namelist():
              CONSOLE.print(f"[Error]Can not find metadata file.", style="bold red")
              return ["META ERROR"]
          else:
              raw_info = zipf.read("info.json").decode("utf-8")
              info = json.loads(raw_info)
              CONSOLE.print(f"[Success]Loaded metadata file.", style="bold green")

          # load the model
          CONSOLE.print("[INFO]Loading model...", style="bold")
          for file in zipf.namelist():
              if file.endswith(".npy"):
                  if file not in info:
                      CONSOLE.print(f"[Error]Can not find file {file} in metadata file.", style="bold red")
                      return ["MODEL ERROR"]
                  self.model[file.split(".")[0]] = np.load(io.BytesIO(zipf.read(file)), allow_pickle=True)
                  # check if the hash of the file is the same as the one in the info.json file
                  if hashlib.sha1(zipf.read(file)).hexdigest() != info[file]:
                      CONSOLE.print("[Error]The file " + file + " is corrupted", style="bold red")
                      return ["HASH ERROR"]
                  CONSOLE.print(f" -Loaded {file}, sha1 = {info[file]}, shape = {self.model[file.split('.')[0]].shape}", style="bold")

          CONSOLE.print("[Success]Model loaded successfully", style="bold green")
          return ["SUCCESS", info]

    def forward(self, array:np.array):
        """
        Operates the model on the given input
        Accepts one board with shape (2, height, width), or a batch with
        shape (batch, 2, height, width).
        """
        batched = array.ndim == 4
        # conv operations
        x = Conv.pooling(Conv.ConvOpe(array, self.model["conv1"]).operation())
        x = ReLU(x)
        x = Conv.pooling(Conv.ConvOpe(x, self.model["conv2"]).operation())
        x = ReLU(x)
        x = x.reshape(x.shape[0], -1) if batched else Conv.view1D(x)
        # linear operations
        i = 0
        while f"w{i}" in self.model:
            x = self.OpeOneLayer(x, self.model[f"w{i}"], self.model[f"b{i}"])
            x = tanh(x) if f"w{i+1}" in self.model else sigmoid(x)
            i += 1

        return x if batched else float(x[0, 0])

    def OpeOneLayer(self, input:np.array, weight:np.array, bias:np.array):
        """
        Operates one layer of the model on the given input
        """
        np.zeros((1, bias.shape[1]))
        return np.matmul(input, weight) + bias