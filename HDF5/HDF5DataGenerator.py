import numpy as np
import h5py

class HDF5DataGenerator:
    def __init__(self, dbPath, data_name, label_name, batch_size, aug = None):
        self.db = h5py.File(dbPath)
        self.batch_size = batch_size
        self.num_samples = self.db[label_name].shape[0]
        self.data_name = data_name
        self.label_name = label_name
        self.aug = aug
    def generator(self, passes = np.inf):
        epoch = 0
        while epoch < passes:
            for i in range(0, self.num_samples, self.batch_size):
                data = self.db[self.data_name][i : i + self.batch_size]
                label = self.db[self.label_name][i : i + self.batch_size]

                # Apply technique about feature engineering
                # Handle:
                # + Scale
                # + Categorical    

                if self.aug:
                    data, label = next(self.aug.flow(data, label, batch_size = self.batch_size))
                yield data, label
            epoch += 1
    def close(self):
        self.db.close()