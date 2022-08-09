import os
import sys
import spacy
import argparse
from colorama import init

def main() -> None:
    """
    Main method, used to parse command line arguments output
    parsed data to the user
    """

    parser = argparse.ArgumentParser(description="Use trained spaCy NER model to parse addresses from a given file")
    parser.add_argument("model", action="store", help="Path to the folder containing the trained model")
    parser.add_argument("--folder", action="store", help="Option to recursively parse data from all files in a given directory")
    parser.add_argument("data", action="store", help="Path to input data that needs to be parsed")
    args = parser.parse_args()

if __name__ == '__main__':
    init()
    main()
