import re
import spacy
import pandas
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
        span = re.sub("\.", '', component)
        span = re.sub(r"(?!\s)(-)(?!\s)", " - ", span)
        span = re.search("\\b(?:" + span + ")\\b", address)
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
    #...
    dataset["EmptySpan"] = dataset.apply(lambda x: [], axis=1)

    for tag in tags:
        dataset["EntitySpans"] = dataset.apply(lambda row: extend(row["EmptySpan"], row[tag]), axis=1)
        dataset["EntitySpans"] = dataset[["EntitySpans", "Address"]].apply(lambda entity: (entity[1], entity[0]), axis=1)
    return dataset["EntitySpans"]

tags = ["Recipient", "Building_Name", "Building_Number", "Street_Name", "City", "State", "Zip_Code", "Country"]
dataset = pandas.read_csv(filepath_or_buffer="./data/datasets/us-train-dataset.csv", sep=",", dtype=str)
entity_spans = create_entity_spans(dataset.astype(str), tags)
print(f"Type: {type(entity_spans)}")
