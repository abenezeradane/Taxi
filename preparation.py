import re
import spacy
import pandas
pandas.set_option("display.max_colwidth", None)

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
    str
        Properly formatted address string
    """

    if pandas.isna(component) or (str(component) == 'nan'):
        pass
    else:
        span = re.sub("\.", '', component)
        span = re.sub(r"(?!\s)(-)(?!\s)", " - ", span)
        span = re.search("\\b(?:" + span + ")\\b", address)
        return (span.start(), span.end(), label)
