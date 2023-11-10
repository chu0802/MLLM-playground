from src.tasks.vqa import VQATask

TASK_DICT = {
    "vqa": VQATask,
}


def get_task(config):
    return TASK_DICT[config.task.name](config)
