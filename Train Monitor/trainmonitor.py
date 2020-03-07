from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt 
import numpy as np
import json
import os 
class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath = None, startAt = 0):
        super().__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = self.startAt
    def on_train_begin(self, logs = {}):
        self.H = {}
        if self.jsonPath:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                if self.startAt:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    def on_epoch_end(self, epoch, logs = {}):
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        if self.jsonPath:
            with open(self.jsonPath, 'w') as f:
                f.write(json.dumps(self.H))
        if len(self.H['loss']) > 1:
            N = range(len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            for label in ('train_loss', 'val_loss', 'train_acc', ' val_acc'):
                plt.plot(N, self.H[label], label = label)
            plt.title(f'Training Loss and Accuracy at Epoch {len(self.H["loss"])}')
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()
