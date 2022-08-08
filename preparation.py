import re
import spacy
import pandas
nlp = spacy.load("en_core_web_trf")
pandas.set_option("display.max_colwidth", None)


def extend(list: list, item: str) -> list:
    """Utility method to safely append an item into a list

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
    """Strips the address string of unnecessary symbols and properly formats
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
    """Return a tuple containing the span of the address component in the
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
    """Create a pandas Series with entity spans for the training dataset

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
    """Return a DocBin (ie. serialization of information) used by spaCy
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
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        doc.ents = ents
        docbin.add(doc)
    return docbin

# Load blank model and define entity tag list
NLP = spacy.blank("en")
TAGS = ["Recipient", "Building_Name", "Building_Number", "Street", "City", "Zip_Code", "Country"]

# Read the training dataset into pandas
DATASET = pandas.read_csv(filepath_or_buffer="./data/datasets/us-train-dataset.csv", sep=",", dtype=str)

# Get entity spans
SPANS = create_entity_spans(DATASET.astype(str), TAGS)
TRAINING_DATA = SPANS.tolist()

# Get and persist DocBin to disk
DOCBIN = create_docbin(TRAINING_DATA, NLP)
DOCBIN.to_disk("./data/docbins/trainer.spacy")
