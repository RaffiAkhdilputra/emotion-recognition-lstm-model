# Emotion Recognition LSTM Model

## Overview

This repository contains a Long Short-Term Memory (LSTM) model for **emotion recognition** from text. The system is designed to classify textual inputs into different emotional categories by training a neural network on labeled datasets.

## Features

* Preprocessing pipeline for text (tokenization, padding, etc.)
* LSTM-based neural network architecture for sequential modeling
* Support for training, evaluation, and saving/loading models
* Sample deployment via Jupyter Notebook

## Repository Structure

```
├── dataset/               # datasets / raw data
├── models/                # trained model files, checkpoints
├── train_model.ipynb      # notebook for training the LSTM model
├── deployment.ipynb       # notebook for running inference / prediction
├── .gitignore  
└── LICENSE  
```

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/RaffiAkhdilputra/emotion-recognition-lstm-model.git
   cd emotion-recognition-lstm-model
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python3 -m venv venv  
   source venv/bin/activate   # On Windows: venv\Scripts\activate  
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *If you don’t have a `requirements.txt`, install typical ML / NLP libraries like `tensorflow`, `numpy`, `pandas`, etc.*

## Usage

### Training

* Open and run `train_model.ipynb` to train the LSTM on your dataset.
* The notebook contains preprocessing steps, model definition, training loop, and evaluation.

### Inference / Deployment

* Use `deployment.ipynb` to run emotion predictions on new text samples.
* Load your trained model, preprocess input text, and run predictions.

## Configuration

* You can modify hyperparameters (e.g. learning rate, batch size, number of LSTM units) directly in the training notebook.
* Adjust the maximum sequence length, embedding size, and number of epochs based on your dataset.

## Dataset

* Place your emotion-labeled text data in the `dataset/` folder.
* Ensure your data is formatted in a way that the training notebook expects (e.g., CSV or JSON with `text` and `label` columns).

## Contributing

Contributions are welcome! Feel free to:

* Improve the preprocessing pipeline
* Try different model architectures (e.g., bidirectional LSTM, GRU)
* Add more emotion categories
* Optimize hyperparameters
* Add evaluation metrics or dashboard

Please fork the repo and submit pull requests, or open issues for discussion.

## License

This project is licensed under the **Unlicense**.

## Contact

Created by **Raffi Akhdilputra**. For questions or collaboration, you can reach me via my GitHub profile.
