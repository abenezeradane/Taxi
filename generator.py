import os
import csv
import sys
import uuid
import argparse

def random_address() -> str:
    """
    Generates a random US address

    Returns
    ----------
    str
        Random US address
    """
    
    pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training dataset to train spaCy address parser NER model")
    parser.add_argument("--output", action="store", help="Filename of generated CSV file")
    args = parser.parse_args()

    if (args.output == None):
        args.output = f"./dataset/{uuid.uuid4()}.csv"

    HEADER = ["Address", "Recipient", "Building_Name", "Building_Number", "Street", "City", "Zip_Code", "Country"]

    with open(args.output, 'w') as FILE:
        WRITER = csv.writer(FILE)
        WRITER.writerow(HEADER)

if __name__ == '__main__':
    main()
