# Tree-Based Machine Learning Algorithms from Scratch

This repository contains Python implementations of several tree-based machine learning algorithms together with comparisons against popular library implementations.

## Installation

Install all required dependencies:

```bash
pip3 install -r requirements.txt
```

Run the notebook or Python scripts after installing the dependencies.

---

## Project Overview

This project demonstrates the implementation and evaluation of several supervised learning algorithms for binary classification.

The project includes:

* Decision Tree Classifier
* Decision Tree Regressor
* Extra Randomized Tree
* Random Forest
* Gradient Boosting Decision Trees (GBDT)

The implementations are written from scratch using NumPy and standard Python without relying on existing tree-based implementations.

In addition, the project compares the custom models with widely used machine learning libraries, including:

* scikit-learn
* LightGBM
* XGBoost
* CatBoost

---

## Dataset

The experiments use a binary classification dataset containing both numerical and categorical features.

The preprocessing pipeline includes:

* chronological train/validation/test split;
* categorical feature encoding;
* handling of previously unseen categorical values;
* feature preparation for different model implementations.

---

## Implemented Algorithms

### Decision Tree

Implemented from scratch with support for:

* Gini impurity criterion;
* recursive tree construction;
* configurable maximum depth;
* probability prediction.

### Decision Tree Regressor

Regression tree implementation using variance reduction as the splitting criterion.

### Extra Randomized Tree

An implementation of extremely randomized trees where candidate split thresholds are sampled randomly before selecting the best split.

### Random Forest

Implementation based on an ensemble of independently trained decision trees with random feature sampling.

### Gradient Boosting Decision Trees

Implementation of gradient boosting using decision trees as weak learners and incremental optimization of binary cross-entropy loss.

---

## Model Evaluation

The project compares:

* custom implementations;
* scikit-learn implementations;
* modern gradient boosting libraries.

The comparison includes training and validation metrics as well as analysis of overfitting and generalization performance.

---

## Technologies

* Python
* NumPy
* pandas
* scikit-learn
* LightGBM
* XGBoost
* CatBoost

---

## Purpose

The purpose of this project is to demonstrate the implementation of classical tree-based machine learning algorithms from scratch, compare them with production-ready libraries, and study their behavior on a real-world classification task.
