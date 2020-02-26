import numpy as np
import h5py
import os
class HDF5DataWriter:
    def __init__(self, outputPath, dims, data_name, label_name, buffer_size = 1024):
        if os.path.exists(outputPath):
            raise ValueError("File Exists!!! Can't overwritten!!!")
        self.dims = dims
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(data_name, dims, dtype = 'float')
        self.label = self.db.create_dataset(label_name, (dims[0], ), dtype = 'int')
        self.buffer_size = buffer_size
        self.buffer = {'data': [], 'label': []}
        self.idx = 0
    def add(self, datas, labels):
        self.buffer['data'].extend(datas)
        self.buffer['label'].extend(labels)
        if len(self.buffer['label']) >= self.buffer_size:
            self.flush()
    def flush(self):
        i                        = self.idx + len(self.buffer['label'])
        self.data [self.idx : i] = self.buffer['data']
        self.label[self.idx : i] = self.buffer['label']
        self.idx = i
        self.buffer = {'data': [], 'label': []}
    def close(self):
        if len(self.data['label']):
            self.flush()
        self.db.close()