from .dataset import Dataset

class DataLoader(object):
    def __init__(self, dataset):
        self.dataset=dataset

    def __iter__(self):
        return self

    def __next__(self):
        db=self.dataset.__getitem__()
        if db is None:
            raise StopIteration
        else:
            return db