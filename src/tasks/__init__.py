from src.tasks.vqa import VQATask

TASK_DICT = {
    "vqa": VQATask,
}


def get_task(task_name):
    return TASK_DICT[task_name]
