

class Summary:
    def __init__(self):
        self.storage = {}

    def add(self, metrics, value):
        self.storage.setdefault(metrics, [])
        if value is not None:
            self.storage[metrics].append(value)

    def get_average(self, metrics):
        lst = self.storage[metrics]
        return None if len(lst) == 0 else sum(lst)/len(lst)

    def get_maximum(self, metrics):
        lst = self.storage[metrics]
        return None if len(lst) == 0 else max(self.storage[metrics])

    def write(self, episode, epsilon):
        s = 'episode: {}, ' \
            'epsilon: {}, ' \
            'average steps: {}, ' \
            'maximum length: {}, ' \
            'average length: {}, ' \
            'average reward: {}, ' \
            'average loss: {}'.format(
            episode,
            epsilon,
            self.get_average('steps'),
            self.get_maximum('length'),
            self.get_average('length'),
            self.get_average('reward'),
            self.get_average('loss')
        )
        with open('logs.txt', 'a') as fout:
            fout.write(s + '\n')
        print(s)

    def clear(self):
        self.storage.clear()