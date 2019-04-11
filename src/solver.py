from dataset import Dataset

class Solver(object):
    def __init__(self, flags):
        self.flags = flags
        self.epoch_time = 0

        # self.dataset = Dataset(self.flags.dataset)

    def train(self):
        print('Hello train!')

    def test(self):
        print('Hello test!')
