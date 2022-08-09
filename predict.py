import os
import sys
import spacy
import argparse
from colorama import init

ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'

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

    if (args.model == None) or (not os.path.exists(args.model)):
        print("\033[91m✘ Model not provided\033[0m")
        sys.exit()

    if (args.data == None) or (not os.path.exists(args.data)):
        print("\033[91m✘ Input not provided\033[0m")
        sys.exit()

    if args.folder == None:
        args.folder = "FILE"

    # Load trained model
    NLP = spacy.load(args.model)

    # Load input data
    CONTENT = []
    if args.folder == "FILE":
        with open(args.data, "r") as FILE:
            CONTENT.append(FILE.readline())

    # Checking predictions for the NER model
    for ADDRESS in CONTENT:
        DOC = NLP(ADDRESS)
        ENTITYLIST = [(ENTITY.text, ENTITY.label_) for ENTITY in DOC.ents]
        print("Address string -> "+ ADDRESS)
        print("Parsed address -> "+str(ENTITYLIST))
        print("******")

if __name__ == '__main__':
    init()
    main()
