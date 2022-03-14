import numpy as np

class DataSet:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        assert X.shape[0] == y.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X = X.copy()
        self.y = y.copy()
        self.indices = np.arange(X.shape[0])

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.X = self.X[self.indices]
        self.y = self.y[self.indices]
        
        for b in range(0, self.X.shape[0], self.batch_size):
            yield self.X[b:b+self.batch_size], self.y[b:b+self.batch_size]