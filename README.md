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
$ python -m spacy download en_core_web_trf

# Prepare the data
$ python preparation.py

# Create training config
$ python -m spacy init fill-config config\base.cfg config\config.cfg

# Train model using data and config
$ python -m spacy train config\config.cfg --output models --paths.train data\docbins\trainer.spacy --paths.dev data\docbins\trainer.spacy
```

### Usage
```bash
usage:

```

### Dependencies
- [spaCy](https://spacy.io)
- [pandas](https://pandas.pydata.org)
