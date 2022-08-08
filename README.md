## Taxi
An address parser implemented in Python using a trained spaCy NER model

### Installation
```bash
# Clone the repo
$ git clone https://github.com/PB020/Taxi.git

# Change the working directory
$ cd taxt

# Create virtual enviroment
$ python -m venv .env
$ .env\Scripts\activate
$ pip install -U pip setuptools wheel

# Install requirements
$ pip install -r requirements.txt

# Prepare the data
$ python preparation.py [DATASET] [TRAINER]

# Create training config
$ python -m spacy init fill-config config\base.cfg [CONFIG]

# Train model using data and config
$ python -m spacy train [CONFIG] --output models --paths.train [TRAINER] --paths.dev [TRAINER] --training.eval_frequency [FREQUENCY] --training.max_steps [STEPS]
```

### Usage
#### `preparation.py`
```bash
usage: preparation.py [-h] dataset trainer

Prepare training dataset to train spaCy address parser NER model

positional arguments:
  dataset     Filename of labeled CSV file used to generate training data
  trainer     Filename of prepared DocBin training data
```

#### `predict.py`
```bash
usage:
```

### Dependencies
- [spaCy](https://spacy.io)
- [pandas](https://pandas.pydata.org)
