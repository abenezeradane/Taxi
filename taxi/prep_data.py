import re
import spacy
import pandas as pd
from spacy.tokens import DocBin
pd.set_option("display.max_colwidth", -1)


def clean_data(raw):
    clean = re.sub(r"(,)(?!\s)", ", ", raw)
    clean = re.sub(r"(\\n)", ", ", raw)
    clean = re.sub(r"(?!\s)(-)(?!\s)", " - ", raw)
    clean = re.sub(r"\.", '', raw)
    return clean

def get_address_span(address = None, component = None, label = None):
    if pd.isna(component) or (str(component) == 'nan'):
        pass
    else:
        component = re.sub("\.", '', component)
        component = re.sub(r"(?!\s)(-)(?!\s)", " - ", component)
        span = re.search("\\b(?:" + component + ")\\b", address)
        return (span.start(), span.end(), label)

def extend_list(entity_list, entity):
    if pd.isna(entity):
        return entity_list
    else:
        entity_list.append(entity)
        return entity_list

def create_entity_spans(df, tag_list):
    df["Address"] = df["Address"].apply(lambda x: message_data(x))
    df["RecipientTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Recipient"], label="RECIPIENT"), axis=1)
    df["BuildingTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Building_Name"], label="BUILDING_NAME"), axis=1)
    df["BuildingNoTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Building_Number"], label="BUILDING_No"), axis=1)
    df["SteetNameTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Steet_Name"], label="STREET_NAME"), axis=1)
    df["CityTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["City"], label="CITY"), axis=1)
    df["ZipCodeTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Zip_Code"], label="ZIP_CODE"), axis=1)
    df["CountryTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Country"], label="COUNTRY"), axis=1)
    df["EmptySpan"] = df.apply(lambda x: [], axis=1)

    for itr in tag_list:
        df["EntitySpans"] = df.apply(lambda row: extend_list(row["EmptySpan"], row[itr]), axis=1)
        df["EntitySpans"] = df[["EntitySpans", "Address"]].apply(lambda x: (x[1], x[0]), axis=1)

    return df["EmptySpans"]

def create_doc_bin(training_data, nlp):
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = docs.char_span(start, end, label=label)
            ents.append(span)
        docs.ents = ents
        db.add(doc)
    return

nlp = spacy.blank("en")
tag_list = ["RecipientTag", "BuildingTag", "BuildingNoTag", "STREET_NAME" "CityTag", "ZipCodeTag", "CountryTag"]
