class DummyVQADataset:
    def __init__(self, size=100):
        self.questions = [f"I am {i}-th question." for i in range(size)]
        self.answers = [
            [f"I am {i}-th answer-{j}." for j in range(10)] for i in range(size)
        ]
        self.image_paths = [f"I am {i}-th image." for i in range(size)]

        pass

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "gt_answers": self.answers[idx],
            "image_path": self.image_paths[idx],
        }
