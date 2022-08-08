import os
import re
import sys
import spacy
import pandas
import argparse
from colorama import init

# Set pandas option
pandas.set_option("display.max_colwidth", None)


def extend(list: list, item: str) -> list:
    """
    Utility method to safely append an item into a list

    Parameters
    ----------
    list: list
        List containing a set of items
    item: str
        String that needs to be appended to the list

    Returns
    ----------
    list
        Original list with the appended item
    """

    if pandas.isna(item) or (str(item) == 'nan'):
        return list
    else:
        list.append(item)
        return list



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

def address_span(address: str = None, component: str = None, label: str = None) -> tuple:
    """
    Return a tuple containing the span of the address component in the
    address string and the classification label of the component

    Parameters
    ----------
    address: str
        String containing the address
    component: str
        String containing the address component
    label: str
        String containing the classification label of the address component

    Returns
    ----------
    tuple
        Tuple of the span of the address component and the classification label
    """

    if pandas.isna(component) or (str(component) == 'nan'):
        pass
    else:
        component = re.sub("\.", "", component)
        component = re.sub(r"(?!\s)(-)(?!\s)", " - ", component)
        span = re.search("\\b(?:" + component + ")\\b", address)
        return (span.start(), span.end(), label)

def create_entity_spans(dataset: pandas.core.frame.DataFrame, tags: list) -> pandas.core.series.Series:
    """
    Create a pandas Series with entity spans for the training dataset

    Parameters
    ----------
    dataset: pandas.core.frame.DataFrame
        pandas DataFrame containing the training dataset
    tags: list
        List of data tags

    Returns
    ----------
    pandas.core.series.Series
        pandas Series of training dataset entity spans
    """

    dataset["Address"] = dataset["Address"].apply(lambda address: strip_address(address))
    dataset["Recipient"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['Recipient'], label='RECIPIENT'), axis=1)
    dataset["Building_Name"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['Building_Name'], label='BUILDING_NAME'), axis=1)
    dataset["Building_Number"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['Building_Number'], label='BUILDING_NUMBER'), axis=1)
    dataset["Street"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['Street_Name'], label='STREET'), axis=1)
    dataset["City"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['City'], label='CITY'), axis=1)
    dataset["State"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['State'], label='STATE'), axis=1)
    dataset["Zip_Code"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['Zip_Code'], label='ZIP_CODE'), axis=1)
    dataset["Country"] = dataset.apply(lambda row: address_span(address=row['Address'], component=row['Country'], label='COUNTRY'), axis=1)
    dataset["EmptySpan"] = dataset.apply(lambda x: [], axis=1)

    for tag in tags:
        dataset["EntitySpans"] = dataset.apply(lambda row: extend(row["EmptySpan"], row[tag]), axis=1)
        dataset["EntitySpans"] = dataset[["EntitySpans", "Address"]].apply(lambda entity: (entity[1], entity[0]), axis=1)
    return dataset["EntitySpans"]

def create_docbin(data: list, NLP: spacy.Language) -> spacy.tokens._serialize.DocBin:
    """
    Return a DocBin (ie. serialization of information) used by spaCy
    as a training set, using training data and an empty spaCy English model

    Parameters
    ----------
    data: list
        List containing training data
    NLP: spacy.Language
        An empty English spaCy model

    Returns
    ----------
    spacy.tokens._serialize.DocBin
        DocBin object for building a training set
    """

    docbin = spacy.tokens.DocBin()
    for text, annotations in data:
        doc = NLP(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        doc.ents = ents
        docbin.add(doc)
    return docbin

ERASE_LINE = '\x1b[2K'
CURSOR_UP_ONE = '\x1b[1A'

def main() -> None:
    """
    Main method, used to parse command line arguments then runs
    training data preparation script
    """

    parser = argparse.ArgumentParser(description="Prepare training dataset to train spaCy address parser NER model")
    parser.add_argument("dataset", action="store", help="Filename of labeled CSV file used to generate training data")
    parser.add_argument("trainer", action="store", help="Filename of prepared DocBin training data")
    args = parser.parse_args()

    if (args.dataset == None) or (not os.path.exists(args.dataset)):
        print("\033[91m✘ Dataset not provided\033[0m")
        sys.exit()

    # Load blank model and add entity ruler
    NLP = spacy.blank("en")
    RULER = NLP.add_pipe("entity_ruler")
    PATTERNS = [
        {"label":"STATE","pattern":[{"LOWER":"alabama"}]},
        {"label":"STATE","pattern":[{"LOWER":"alaska"}]},
        {"label":"STATE","pattern":[{"LOWER":"arizona"}]},
        {"label":"STATE","pattern":[{"LOWER":"arkansas"}]},
        {"label":"STATE","pattern":[{"LOWER":"california"}]},
        {"label":"STATE","pattern":[{"LOWER":"colorado"}]},
        {"label":"STATE","pattern":[{"LOWER":"connecticut"}]},
        {"label":"STATE","pattern":[{"LOWER":"delaware"}]},
        {"label":"STATE","pattern":[{"LOWER":"florida"}]},
        {"label":"STATE","pattern":[{"LOWER":"georgia"}]},
        {"label":"STATE","pattern":[{"LOWER":"hawaii"}]},
        {"label":"STATE","pattern":[{"LOWER":"idaho"}]},
        {"label":"STATE","pattern":[{"LOWER":"illinois"}]},
        {"label":"STATE","pattern":[{"LOWER":"indiana"}]},
        {"label":"STATE","pattern":[{"LOWER":"iowa"}]},
        {"label":"STATE","pattern":[{"LOWER":"kansas"}]},
        {"label":"STATE","pattern":[{"LOWER":"kentucky"}]},
        {"label":"STATE","pattern":[{"LOWER":"louisiana"}]},
        {"label":"STATE","pattern":[{"LOWER":"maine"}]},
        {"label":"STATE","pattern":[{"LOWER":"maryland"}]},
        {"label":"STATE","pattern":[{"LOWER":"massachusetts"}]},
        {"label":"STATE","pattern":[{"LOWER":"michigan"}]},
        {"label":"STATE","pattern":[{"LOWER":"minnesota"}]},
        {"label":"STATE","pattern":[{"LOWER":"mississippi"}]},
        {"label":"STATE","pattern":[{"LOWER":"missouri"}]},
        {"label":"STATE","pattern":[{"LOWER":"montana"}]},
        {"label":"STATE","pattern":[{"LOWER":"nebraska"}]},
        {"label":"STATE","pattern":[{"LOWER":"nevada"}]},
        {"label":"STATE","pattern":[{"LOWER":"new hampshire"}]},
        {"label":"STATE","pattern":[{"LOWER":"new jersey"}]},
        {"label":"STATE","pattern":[{"LOWER":"new mexico"}]},
        {"label":"STATE","pattern":[{"LOWER":"new york"}]},
        {"label":"STATE","pattern":[{"LOWER":"north carolina"}]},
        {"label":"STATE","pattern":[{"LOWER":"north dakota"}]},
        {"label":"STATE","pattern":[{"LOWER":"ohio"}]},
        {"label":"STATE","pattern":[{"LOWER":"oklahoma"}]},
        {"label":"STATE","pattern":[{"LOWER":"oregon"}]},
        {"label":"STATE","pattern":[{"LOWER":"pennsylvania"}]},
        {"label":"STATE","pattern":[{"LOWER":"rhode island"}]},
        {"label":"STATE","pattern":[{"LOWER":"south carolina"}]},
        {"label":"STATE","pattern":[{"LOWER":"south dakota"}]},
        {"label":"STATE","pattern":[{"LOWER":"tennessee"}]},
        {"label":"STATE","pattern":[{"LOWER":"texas"}]},
        {"label":"STATE","pattern":[{"LOWER":"utah"}]},
        {"label":"STATE","pattern":[{"LOWER":"vermont"}]},
        {"label":"STATE","pattern":[{"LOWER":"virginia"}]},
        {"label":"STATE","pattern":[{"LOWER":"washington"}]},
        {"label":"STATE","pattern":[{"LOWER":"west virginia"}]},
        {"label":"STATE","pattern":[{"LOWER":"wisconsin"}]},
        {"label":"STATE","pattern":[{"LOWER":"wyoming"}]}
    ]
    RULER.add_patterns(PATTERNS)

    # Read the training dataset into pandas
    print("\033[92m✔ Fetching dataset\033[0m")
    DATASET = pandas.read_csv(filepath_or_buffer=args.dataset, sep=",", dtype=str)
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
    print("\033[92m✔ Fetched dataset\033[0m")

    # Define tags and get entity spans
    TAGS = ["Recipient", "Building_Name", "Building_Number", "Street", "City", "Zip_Code", "Country"]
    SPANS = create_entity_spans(DATASET.astype(str), TAGS)
    TRAINING_DATA = SPANS.tolist()

    # Get and persist DocBin to disk
    print("\033[92m✔ Creating spaCy training data\033[0m")
    DOCBIN = create_docbin(TRAINING_DATA, NLP)
    DOCBIN.to_disk(args.trainer)
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
    print("\033[92m✔ Created spaCy training data\033[0m")
    print("\033[97mReady to start training models!\033[0m")

if __name__ == '__main__':
    init()
    main()
