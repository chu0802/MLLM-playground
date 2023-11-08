class DummyModel:
    def __init__(self):
        pass

    def eval(self):
        pass

    def batch_generate(self, image_paths, questions, max_len, min_len, num_beams):
        return range(len(image_paths))
