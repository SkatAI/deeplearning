# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a 3-day deep learning training course ("Deep Learning par la pratique") by SkatAI, using TensorFlow/Keras. The course is in French. It covers: perceptron, MLP, CNN, RNN/LSTM, autoencoders, and NLP (BERT).

## Running Code

All notebooks are designed to run on **Google Colab with a GPU runtime**. Many `src/` Python files are Colab-exported scripts (contain `!wget`, `%load_ext` magic commands) and are not meant to run locally as-is.

To run a notebook: open it in Google Colab, set runtime to GPU, and execute cells.

There is no requirements.txt, Makefile, or test suite. Key dependencies are TensorFlow, Keras, scikit-learn, matplotlib, plotly, pandas, numpy, and (for NLP) transformers + BeautifulSoup.

## Architecture

- **`notebooks/`** — Jupyter notebooks, the primary teaching material. Each notebook corresponds to a course session topic (perceptron, CNN, RNN, transfer learning, autoencoders, BERT, etc.)
- **`src/`** — Python scripts auto-exported from Colab notebooks. These mirror the notebook content but as `.py` files. They are not standalone scripts; many contain Colab-specific shell commands (`!wget`, `!ls`)
- **`data/`** — Small local datasets (CSV files, text corpora for RNN/text generation: Moliere, Maupassant)
- **`slides/`** — PDF slide decks for each half-day session (J1 AM/PM, J2 AM/PM, J3 AM/PM)
- **`docs/`** — Reference papers and notes (attention, dropout, deep learning overview)
- **`models/`** — Saved model archives
- **`img/`** — Images used in slides/notebooks

## Course Structure (6 sessions)

1. **S1** — ML refresher, SGD, perceptron
2. **S2** — MLP, TensorFlow/Keras ecosystem, tensors, backpropagation
3. **S3** — Callbacks, CNN, TensorBoard, transfer learning (InceptionV3)
4. **S4** — RNN, LSTM, GRU, time series
5. **S5** — Autoencoders, denoising autoencoders, VAEs
6. **S6** — NLP, BERT sentiment classification

## Conventions

- Notebooks and code comments are in French
- Datasets are either loaded from Keras built-ins (MNIST, Fashion-MNIST, CIFAR, cats_and_dogs) or from `data/` (housing.csv, sunspots.csv, moliere.txt)
- Some notebooks download datasets/weights at runtime via `wget` (e.g., cats_and_dogs_filtered, InceptionV3 weights)
