from src.datasets.vqa import BaseMultiAnswersVQADataset
import json


class OKVQATrainDataset(BaseMultiAnswersVQADataset):
    def load_data(self):
        data = json.load(
            (self.data_root / "annotations" / "okvqa_train.json").open("r")
        )
        image_dir = self.data_root / "images"

        for d in data:
            self.image_path_list.append((image_dir / d["image"]).as_posix())
            self.question_list.append(d["question"])
            self.answer_weight_list.append(self.parse_answer(d["answer"]))


class OKVQAEvalDataset(BaseMultiAnswersVQADataset):
    def load_data(self):
        questions = json.load(
            (
                self.data_root
                / "annotations"
                / "OpenEnded_mscoco_val2014_questions.json"
            ).open("r")
        )["questions"]
        annotations = json.load(
            (self.data_root / "annotations" / "mscoco_val2014_annotations.json").open(
                "r"
            )
        )["annotations"]

        question_dict = {x["question_id"]: x["question"] for x in questions}

        for i in range(len(annotations)):
            question = question_dict[annotations[i]["question_id"]]
            answers = [x["answer"] for x in annotations[i]["answers"]]
            image_path = (
                self.data_root
                / "images"
                / "val2014"
                / f"COCO_val2014_{annotations[i]['image_id']:012d}.jpg"
            )

            self.image_path_list.append(image_path.as_posix())
            self.question_list.append(question)
            self.answer_weight_list.append({"answers": answers})


if __name__ == "__main__":
    from src.utils.config import get_config

    config = get_config()
    config.dataset.name = "OKVQA"
    config.dataset.split.eval.name = "val"
    ok_vqa = OKVQAEvalDataset(config, split=config.dataset.split.eval.name)

    l = ok_vqa[0]
    pass
