# VLM-MODEL
# VCR Object-Aware Visual Reasoning

A Visual Commonsense Reasoning (VCR) project using **Qwen2.5-VL-3B-Instruct** to investigate whether explicitly providing information about objects referenced in a question can improve visual question answering and rationale selection.

## Overview

Visual Commonsense Reasoning requires a model to understand an image, answer a multiple-choice question about the scene, and identify the rationale that best supports the answer.

This project explores an **object-aware approach** to VCR. Instead of relying only on the original image, the experiment identifies objects referenced by the question and uses their corresponding image regions as additional visual information.

The project compares:

1. **Baseline:** Qwen2.5-VL-3B-Instruct receives the original image and VCR question.
2. **Object-aware approach:** Referenced objects are identified and their corresponding regions are used to provide additional visual information during answer selection.

## Objectives

* Understand the structure of the VCR dataset.
* Build a baseline visual reasoning system using a vision-language model.
* Identify objects referenced in VCR questions.
* Extract object regions using bounding-box information.
* Compare standard image-based reasoning with object-aware reasoning.
* Extend the system to rationale selection.
* Evaluate answer, rationale, and joint answer-rationale performance.

## Dataset

The project uses the **VCR (Visual Commonsense Reasoning)** dataset through the Hugging Face dataset:

`Rowan/vcr`

The notebook loads the `questions` configuration and uses the training split in streaming mode.

Each VCR example contains information including:

* Image information
* Detected objects
* Object bounding boxes
* Question
* Four answer choices
* Correct answer label
* Four rationale choices
* Correct rationale label

The dataset also contains references to specific objects inside questions and answer/rationale choices.

## Model

The main vision-language model used in this project is:

**Qwen2.5-VL-3B-Instruct**

The model is loaded using the Hugging Face Transformers library and is used for both baseline visual question answering and the object-aware experiment.

## Methodology

### 1. VCR Data Processing

The notebook first loads the VCR dataset and examines its structure, including:

* Questions
* Answer choices
* Rationale choices
* Object information
* Object references
* Ground-truth labels

### 2. Object Reference Extraction

VCR questions can contain references such as `[person2]`.

The project identifies the object indices referenced by the question and maps these references to the corresponding objects in the image.

### 3. Object Localization

The corresponding bounding-box information is used to locate the referenced objects.

The project visualizes these bounding boxes on the original image and extracts the relevant object regions.

### 4. Baseline

For the baseline experiment, Qwen2.5-VL receives:

* The original image
* The VCR question
* Four candidate answers

The model is prompted to select the single best answer and return its answer index.

### 5. Object-Aware Experiment

The object-aware experiment incorporates the referenced objects identified from the VCR question.

For each candidate answer, the model evaluates whether the candidate is supported by the visual evidence. Candidate scores are then compared to select the predicted answer.

This allows the project to investigate whether explicitly highlighting question-relevant visual information helps the model perform VCR reasoning.

### 6. Rationale Selection

After answer prediction, the system evaluates the four candidate rationales.

The model is asked to determine whether each rationale is the best explanation for the selected answer based on:

* Visual evidence
* Logical connection to the question
* Consistency with the image

The candidate with the highest score is selected as the predicted rationale.

## Evaluation

The project evaluates the system on subsets of the VCR data.

### Answer Evaluation

A 50-example evaluation was performed comparing the baseline and object-aware approaches.

The recorded notebook results include:

| Approach                | Accuracy |
| ----------------------- | -------: |
| Baseline                |      66% |
| Object-aware experiment |  62–64%* |

*The notebook contains results from two evaluation executions (64% in one run and 62% in a later validation output). A final single evaluation run should be used as the authoritative result before submission.

Both approaches produced valid predictions for all 50 evaluated examples in the later validation output.

### Joint Answer and Rationale Evaluation

An end-to-end evaluation was performed on 20 examples.

| Metric                            | Accuracy |
| --------------------------------- | -------: |
| Answer accuracy                   |      60% |
| Rationale accuracy                |      35% |
| Answer + rationale joint accuracy |      25% |

The joint evaluation measures the full pipeline, where the predicted answer is passed to the rationale-selection stage.

### Isolated Rationale Evaluation

A separate diagnostic experiment evaluated rationale selection while supplying the **ground-truth answer** to the rationale model.

This produced:

**Rationale accuracy: 45%**

This result is treated as an auxiliary diagnostic rather than the end-to-end rationale result, since the ground-truth answer is supplied instead of the model's predicted answer.

## Results and Observations

The current experiments do not show an improvement in answer accuracy from the object-aware approach over the baseline on the evaluated subset.

This suggests that simply providing additional information about referenced objects does not necessarily improve VCR reasoning with the current model and prompting strategy.

The result is useful because it highlights a limitation of straightforward object-level visual grounding: identifying and supplying relevant object regions alone may not be sufficient for commonsense reasoning.

The relatively lower rationale and joint accuracy also indicate that selecting the correct answer and selecting a rationale that correctly supports that answer are distinct challenges.


## Repository Structure

```text
VCR-Object-Aware-Reasoning/
│
├── VCR_Project.ipynb
├── README.md
├── requirements.txt
└── results/
    └── comparison.csv
```

## Requirements

The project uses Python and the following major libraries:

* PyTorch
* Transformers
* Hugging Face Datasets
* Accelerate
* Qwen-VL utilities
* Pillow
* Pandas
* Matplotlib

Install the main dependencies with:

```bash
pip install datasets transformers accelerate
pip install qwen-vl-utils
```

The notebook also installs the latest Transformers version directly from the Transformers GitHub repository when required by the Qwen2.5-VL implementation.

## Running the Project

1. Open `VCR_Project.ipynb`.
2. Install the required dependencies.
3. Load the VCR dataset.
4. Load Qwen2.5-VL-3B-Instruct.
5. Run the data inspection and object-reference processing cells.
6. Run the baseline evaluation.
7. Run the object-aware evaluation.
8. Run the rationale evaluation.
9. Run the final results cells.

A CUDA-enabled environment is recommended because the vision-language model is computationally intensive.

## Conclusion

This project investigates whether explicitly grounding a vision-language model in objects referenced by a VCR question can improve visual commonsense reasoning.

The experiments demonstrate a complete pipeline from VCR data processing and object localization to answer prediction and rationale selection. While the current object-aware approach does not outperform the baseline on the evaluated subset, the results provide a useful basis for investigating more sophisticated object-centric reasoning methods.

---

## Author

**Dakshita Malik**

