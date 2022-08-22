import uuid
import random
import argparse
from sys import exit

def n(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training dataset to train spaCy address parser NER model")
    parser.add_argument("input", action="store", help="Original datastset to augment")
    parser.add_argument("--output", action="store", help="Filename of generated CSV file")
    args = parser.parse_args()

    if (args.input == None):
        exit()

    if (args.output == None):
        args.output = f"./dataset/{uuid.uuid4()}.csv"

    DATASET = []
    with open(args.input, "r") as f:
        for line in f:
            DATASET.append(line)
    DATASET.pop(0)

    HEADER = ["Address", "Building_Name", "Building_Number", "City", "Recipient", "Street_Name", "Zip_Code", "State", "Country"]
    with open(args.output, 'w') as FILE:
        FILE.write(f"{','.join(HEADER)}\n")
        for DATA in DATASET:
            FILETYPE = random.choice(["Apple Binary", "ASCII Text"])
            if FILETYPE == "Apple Binary":
                ADR = DATA.split("\"")[1]
                DATA = f"\"<string>{ADR}</string>" + DATA[len(ADR) + 1:]
                FILE.write(DATA)
            elif FILETYPE == "ASCII Text":
                FILE.write(DATA)

if __name__ == '__main__':
    main()
