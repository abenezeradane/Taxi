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
