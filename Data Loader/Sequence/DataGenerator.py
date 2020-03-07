import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X, label, batch_size = 32):
        self.X = X
        self.label = label
        self.batch_size = batch_size
        self.num_samples = self.X[0]
    def __len__(self):
        return self.num_samples // self.batch_size
    def __getitem__(self, index):
        batch_X = self.X[index * self.batch_size : (index + 1) * self.batch_size]
        batch_label = self.label[index * self.batch_size : (index + 1) * self.batch_size]

        # Apply technique about feature engineering
        # Handle:
        # + Scale
        # + Categorical    

        return batch_X, batch_label