# MLLM-playground
This is a multi-modal large language model playground, way more than a benchmark.

## Overview

MLLM-playground, short for Multimodal Large Language Model Playground, is a toolkit designed to streamline the training and evaluation processes for various vision-and-language datasets using different multimodal large language models. This project offers a unified and user-friendly interface to facilitate the experimentation and development of multimodal models.

## Installation
Follow these steps to set up MLLM-playground on your local machine.

### Prerequisites

* Python >= 3.10

### Installation

1. Clone the repository, and navigate to the project directory.
   
   ```bash
   git clone https://github.com/chu0802/MLLM-playground.git
   cd MLLM-playground
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train and evaluate models using MLLM-playground, two main scripts are provided:
* `train.py`
* `eval.py`

Before running these scripts, it's essential to set up the configuration files appropriately.


### Configuration Setup

* Training Configuration:
  * Open `train_config.yaml` and adjust the parameters according to your experimental setup. Specify the dataset, model architecture, hyperparameters, and any other relevant settings.

* Evaluation Configuration:
  * Similarly, in `eval_config.yaml`, configure the parameters needed for evaluation, such as the path to the trained model, evaluation metrics, etc.


### Training

Run the training script using the following command:

```bash
python train.py --cfg-path train_config.yaml
```

We provide basic settings in `train_config.yaml`. But you can overwrite specific settings in the `train_config.yaml` file by adding command-line options. For example:

```bash
python train.py --cfg-path train_config.yaml --options dataset.name=ScienceQA dataset.split.train.batch_size=16
```

This command will overwrite the dataset and batch size for training specified in the configuration file.

### Evaluation

After training, you can evaluate the model by running the evaluation script:

```bash
python eval.py --cfg-path eval_config.yaml
```

Similarly, ensure that the evaluation configuration in `eval_config.yaml` is appropriately set up for your experiment, and you can also overwrite the settings by specifying `--options` arguments.


### Monitoring

During training, we can monitor the progress on the [WandB](http://wandb.ai) dashboard. The trained model will be saved for every epoch according to the settings in the configuration file.

By following these steps, you can efficiently train and evaluate multimodal large language models on various datasets using MLLM-playground. :relaxed:

## Acknowledgments

This code base is partially based on **LVLM-eHub** [[paper](https://arxiv.org/abs/2306.09265), [code](https://github.com/OpenGVLab/Multi-Modality-Arena)] and **Lavis** [[paper](https://arxiv.org/abs/2209.09019), [code](https://github.com/salesforce/LAVIS)].
