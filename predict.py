import os
import re
import sys
import spacy
import argparse
from colorama import init

def strip_address(address: str) -> str:
    """
    Strips the address string of unnecessary symbols and properly formats
    the address into a csv file style format using regex

    Parameters
    ----------
    address: str
        String containing the address

    Returns
    ----------
    str
        Properly formatted address string
    """

    stripped = re.sub(r"(,)(?!\s)", ", ", address)
    stripped = re.sub(r"(\\n)", ", ", stripped)
    stripped = re.sub(r"(?!\s)(-)(?!\s)", " - ", stripped)
    stripped = re.sub(r"\.", "", stripped)
    return stripped

def parse_address(nlp: spacy.Language, address: str, output: str) -> list:
    """
    Parses the passed address string and returns the address components
    as a list of tuples

    Parameters
    ----------
    NLP: spacy.Language
        An empty English spaCy model
    address: str
        String containing the address
    output: str
        String containing the filename to save data output

    Returns
    ----------
    List
        List of address components
    """

    output = open(output, "a+")
    doc = nlp(strip_address(address))
    entities = [(entity.text, entity.label_) for entity in doc.ents]

    output.write(f"Address: {address[0:-1]}\n")
    print(f"\033[94mAddress:\033[0m \033[97m{address[0:-1]}\033[0m")
    for entity in entities:
        output.write(f"  {entity[1]}: {entity[0]}\n")
        print(f"  \033[94m{entity[1]}:\033[0m \033[97m{entity[0]}\033[0m")
    output.write("\n")
    print("")

    output.close()
    return entities

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
    parser.add_argument("--output", action="store", help="Filename to save parsed data")
    args = parser.parse_args()

    if (args.model == None) or (not os.path.exists(args.model)):
        print("\033[91m✘ Model not provided\033[0m")
        sys.exit()

    if (args.data == None) or (not os.path.exists(args.data)):
        print("\033[91m✘ Input not provided\033[0m")
        sys.exit()

    if args.output == None:
        args.output = "./output.txt"

    if args.folder == None:
        args.folder = "FILE"

    # Load trained model
    NLP = spacy.load(args.model)

    # Load input data
    CONTENT = []
    if args.folder == "FILE":
        FILE = open(args.data, "r")
        CONTENT = FILE.readlines()
    else:
        FILES = next(os.walk(args.data), (None, None, []))[2]
        for file in FILES:
            PATH = f"{args.data}\{file}"
            DATA = open(PATH, "r")
            CONTENT += DATA.readlines()

    # Checking predictions for the NER model
    for ADDRESS in CONTENT:
        parse_address(NLP, ADDRESS, args.output)

if __name__ == '__main__':
    init()
    main()
