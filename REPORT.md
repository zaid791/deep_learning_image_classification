# Image Classification on CINIC-10 Using Convolutional Neural Networks

## Abstract

This report investigates image classification on the CINIC-10 dataset with convolutional neural networks and related deep learning methods. The objective is to compare several model architectures, assess the influence of selected training and regularization hyperparameters, evaluate the effect of data augmentation strategies, and examine performance under reduced-data and few-shot settings. In addition to predictive performance, the study emphasizes reproducibility, statistical summary of repeated runs, and practical considerations related to limited computing resources. The work is conducted as a two-person project, with GPU-intensive experiments delegated to the teammate's machine and lighter experiments remaining feasible on a MacBook Air M2 where possible.

## Table of Contents

1. [Research Problem](#research-problem)
2. [Theoretical Introduction and Literature Review](#theoretical-introduction-and-literature-review)
3. [Experimental Methodology](#experimental-methodology)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Conclusions and Further Work](#conclusions-and-further-work)
7. [Application Instructions and Reproducibility](#application-instructions-and-reproducibility)
8. [Division of Work and Computational Constraints](#division-of-work-and-computational-constraints)
9. [Bibliography](#bibliography)

## Research Problem

The research problem addressed in this project is image classification for the CINIC-10 dataset, which contains 10 object categories and presents a more challenging benchmark than the original CIFAR-10 test set due to its construction from both ImageNet and CIFAR-10 imagery. The task is to determine which neural network architectures and training strategies provide the best trade-off between accuracy, stability, and computational cost under realistic resource constraints.

The study is designed to answer the following questions:

1. How do different network architectures compare on CINIC-10?
2. How sensitive are the results to selected training hyperparameters and regularization choices?
3. Which data augmentation methods improve generalization most clearly?
4. Can a simple few-shot method remain competitive when the training set is reduced substantially?
5. How much does model quality change when the training set is reduced?
6. Which methods are feasible on limited hardware, and which require GPU acceleration?

## Theoretical Introduction and Literature Review

### Convolutional Neural Networks

Convolutional neural networks (CNNs) are the standard architecture for image recognition tasks because they exploit spatial locality, parameter sharing, and hierarchical feature learning. Early layers typically learn edges and textures, while deeper layers learn class-specific patterns. This inductive bias makes CNNs well suited for small to medium-sized image classification datasets.

### Transfer Learning and Pretrained Models

Modern vision models often use transfer learning, in which a model pretrained on a large dataset such as ImageNet is adapted to a smaller target task. This approach is particularly useful when the target dataset is limited or when training time is constrained. In this project, pretrained models are considered as allowed and recommended by the assignment brief.

### Regularization and Generalization

Regularization methods such as dropout, weight decay, label smoothing, and data augmentation reduce overfitting by limiting model capacity or increasing the effective diversity of the training data. Their influence is especially important for relatively small image datasets and for experiments that intentionally reduce the training set.

### Data Augmentation

Data augmentation increases the variability of the training set through label-preserving transformations. Standard augmentations include random crops, flips, rotations, and color perturbations. More advanced techniques such as Cutout, MixUp, CutMix, or AutoAugment often provide additional improvements by exposing the model to more difficult training examples.

### Few-Shot Learning

Few-shot learning methods aim to achieve reasonable performance when only a small number of examples per class is available. Practical approaches include transfer learning with a frozen feature extractor, prototypical classification in embedding space, and nearest-centroid methods. Because the assignment requires one dedicated few-shot method, a lightweight and reproducible solution will be implemented and compared with the full-data baseline.

### Relevant References

The bibliography will include core references on CNNs, transfer learning, regularization, and augmentation methods used in the experiments. Exact citations will be added after the final experiment set is confirmed.

## Experimental Methodology

### Dataset

The CINIC-10 dataset is used exclusively, in accordance with the project restrictions. The data are split into training, validation, and test subsets. The main preprocessing step is channel normalization using the CINIC-10 mean and standard deviation.

### Models to Compare

Planned model families include:

1. A custom baseline CNN.
2. A pretrained reference model, such as ResNet18 or EfficientNet-B0.
3. Optional additional variants if time and compute allow.

### Training Hyperparameters

At least two training-related hyperparameters will be investigated, for example:

1. Learning rate.
2. Batch size.
3. Optimizer choice.
4. Learning-rate schedule.

### Regularization Hyperparameters

At least two regularization-related hyperparameters will be investigated, for example:

1. Weight decay.
2. Dropout rate.
3. Label smoothing.
4. Strength or probability of augmentation.

### Data Augmentation Experiments

The report will include at least three standard augmentations and one advanced augmentation method. Planned candidates include:

1. Random horizontal flip.
2. Random crop or resized crop.
3. Color jitter or rotation.
4. Cutout, CutMix, MixUp, RandAugment, or AutoAugment.

### Few-Shot and Reduced-Data Experiments

A dedicated few-shot method will be evaluated on a reduced training subset. In addition, the performance of models trained on smaller fractions of the dataset will be compared against the full-data baseline. This comparison is important because the assignment explicitly asks for reduced-data analysis and because limited hardware may require smaller runs.

### Repeated Runs and Statistical Summary

To support statistically meaningful conclusions, each key experiment should be repeated multiple times when feasible. For every reported metric, the mean and standard deviation will be calculated across runs. If compute limitations prevent full repetition for all experiments, this limitation will be documented explicitly in the report.

## Results

This section will be populated after the experiments are run. Each experiment should be reported with a clear description of the setting, a table of metrics, and commentary on the observed behavior.

### Result Table Template

| Experiment | Model | Augmentation | Regularization | Train Fraction | Validation Accuracy | Test Accuracy | Mean +/- Std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline-1 | To be filled | To be filled | To be filled | 100% | To be filled | To be filled | To be filled |
| Baseline-2 | To be filled | To be filled | To be filled | 100% | To be filled | To be filled | To be filled |
| Few-shot | To be filled | To be filled | To be filled | Reduced | To be filled | To be filled | To be filled |

### Figures

Planned figures include training and validation loss curves, accuracy curves, and a comparison plot showing the effect of reduced training data. All figures should be referenced in the text and include captions.

## Discussion

This section will interpret the results rather than merely listing them. The discussion should address:

1. Which architecture performed best and why.
2. Which augmentation methods helped generalization most.
3. Whether the regularization settings improved stability or reduced overfitting.
4. How much performance was lost under few-shot or reduced-data conditions.
5. Whether any surprising failures occurred and what they suggest about the model or setup.
6. Whether limited compute prevented some experiments from being repeated or expanded.

## Conclusions and Further Work

The conclusion will summarize the main findings of the project and assess whether the chosen methods were effective under the available compute budget. It should also state the most likely reasons for success or failure and propose reasonable next steps, such as trying stronger pretrained backbones, better augmentation policies, cross-validation, or a larger sweep of hyperparameters.

## Application Instructions and Reproducibility

### Environment Setup

1. Create a Python virtual environment.
2. Install dependencies from the appropriate requirements file.
3. Download the CINIC-10 dataset and place it in `data/cinic10/`.

### Running the Project

The current repository includes a dataset-loading entrypoint. As the implementation is extended, reproducible commands for training, evaluation, and experiment replication will be documented here.

### Reproducibility Controls

The report and code should record:

1. Fixed random seed.
2. Model architecture and hyperparameters.
3. Data augmentation settings.
4. Train/validation/test split handling.
5. Hardware used for each run.
6. Number of repetitions per experiment.

### Expected Hardware Notes

Some experiments may be practical on the MacBook Air M2, especially dataset loading, lightweight baselines, and small-scale checks. GPU-enabled training on the teammate's machine should be used for larger sweeps, pretrained models, repeated runs, and the more expensive augmentation experiments.

## Division of Work and Computational Constraints

This project is a two-person effort. The report should clearly state which parts were performed on each machine to make the workflow transparent and reproducible.

Suggested division:

1. MacBook Air M2: code preparation, dataset verification, lightweight smoke tests, and smaller baseline runs where feasible.
2. GPU machine: full training runs, repeated experiments, pretrained model comparisons, hyperparameter sweeps, and statistically summarized results.

If a required experiment cannot be run at full scale due to limited compute, the limitation should be stated explicitly, along with the mitigation strategy used. Acceptable mitigations include reducing the number of samples, reducing image resolution, removing selected classes for a controlled pilot study, or lowering the number of repetitions.

## Bibliography

The final report should include a bibliography with all referenced sources. At minimum, it should cover the CINIC-10 dataset description, core CNN literature, transfer learning references, and the augmentation or few-shot methods actually used in the experiments.

Possible reference categories to include:

1. Dataset paper or dataset documentation for CINIC-10.
2. Foundational CNN references.
3. Transfer learning references.
4. Data augmentation references.
5. Few-shot learning references.
6. Any papers or documentation for pretrained architectures used in the experiments.

---

## Notes for Future Updates

- Replace placeholder text with actual experiment descriptions and measured values.
- Add figure files and link them from the Results section.
- Add precise citations once the final set of experiments is fixed.
- Record the final random seed and command lines used for each run.
