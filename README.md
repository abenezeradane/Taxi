## Taxi
An address parser implemented in Python using a trained spaCy NER model

### Installation
```bash
# Clone the repo
$ git clone https://github.com/PB020/Taxi.git

# Change the working directory
$ cd taxi

# Create virtual enviroment
$ python -m venv .env
$ .env\Scripts\activate
$ pip install -U pip setuptools wheel

# Install requirements
$ pip install -r requirements.txt

# Prepare the data
$ python preparation.py data\datasets\us-train-dataset.csv [TRAINER]

# Create training config
$ python -m spacy init fill-config config\base.cfg config\config.cfg

# Train model using data and config
$ python -m spacy train [CONFIG] --output models --paths.train [TRAINER] --paths.dev [TRAINER]
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
usage: predict.py [-h] [--folder FOLDER] [--output OUTPUT] model data

Use trained spaCy NER model to parse addresses from a given file

positional arguments:
  model            Path to the folder containing the trained model
  data             Path to input data that needs to be parsed

optional arguments:
  -h, --help       show this help message and exit
  --folder FOLDER  Option to recursively parse data from all files in a given directory
  --output OUTPUT  Filename to save parsed data
```

#### Example Usage
```bash
# Single File Usage
$ python predict.py "models\model-best" "data\tests\us-test-dataset.csv"

# Directory Usage
$ python predict.py "models\model-best" --folder FOLDER "data\tests" --output "output\test.txt"
```

### Dependencies
- [spaCy](https://spacy.io)
- [pandas](https://pandas.pydata.org)
