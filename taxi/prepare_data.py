import re
import spacy
import pandas as pd
from spacy.tokens import DocBin
pd.set_option("display.max_colwidth", -1)


def clean_data(raw):
    '''Use regex to pre-emptively clean the data (remove new line characters, proper csv punctuation, etc.)'''
    clean = re.sub(r"(,)(?!\s)", ", ", raw)
    clean = re.sub(r"(\\n)", ", ", clean)
    clean = re.sub(r"(?!\s)(-)(?!\s)", " - ", clean)
    clean = re.sub(r"\.", '', clean)
    return clean

def get_address_span(address = None, component = None, label = None):
    '''Search for a specified component and get the span'''
    if pd.isna(component) or (str(component) == 'nan'):
        pass
    else:
        component = re.sub("\.", '', component)
        component = re.sub(r"(?!\s)(-)(?!\s)", " - ", component)
        span = re.search("\\b(?:" + component + ")\\b", address)
        return (span.start(), span.end(), label)

def extend_list(entity_list, entity):
    '''Controlled list append'''
    if pd.isna(entity):
        return entity_list
    else:
        entity_list.append(entity)
        return entity_list

def create_entity_spans(df, tag_list):
    '''Create entity spans for training datasets'''
    df["Address"] = df["Address"].apply(lambda x: message_data(x))
    df["RecipientTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Recipient"], label="RECIPIENT"), axis=1)
    df["BuildingTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Building_Name"], label="BUILDING_NAME"), axis=1)
    df["BuildingNoTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Building_Number"], label="BUILDING_No"), axis=1)
    df["SteetNameTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Steet_Name"], label="STREET_NAME"), axis=1)
    df["CityTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["City"], label="CITY"), axis=1)
    df["StateTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["State"], label="STATE"), axis=1)
    df["ZipCodeTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Zip_Code"], label="ZIP_CODE"), axis=1)
    df["CountryTag"] = df.apply(lambda row: get_address_span(address=row["Address"], component=["Country"], label="COUNTRY"), axis=1)
    df["EmptySpan"] = df.apply(lambda x: [], axis=1)

    for itr in tag_list:
        df["EntitySpans"] = df.apply(lambda row: extend_list(row["EmptySpan"], row[itr]), axis=1)
        df["EntitySpans"] = df[["EntitySpans", "Address"]].apply(lambda x: (x[1], x[0]), axis=1)

    return df["EmptySpans"]

def create_doc_bin(training_data, nlp):
    '''Create DocBin object for building training corpus'''
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

'''Load blank model with custom entity tag list'''
nlp = spacy.blank("en")
tag_list = ["RecipientTag", "BuildingTag", "BuildingNoTag", "STREET_NAME" "CityTag", "ZipCodeTag", "CountryTag"]

########## Training dataset preparation script ##########
df_train = pd.read_csv(filepath_or_buffer="../data/datasets/us-train-dataset.csv", sep=",", dtype=str)

df_entity_spans = create_entity_spans(df_train.astype(str), tag_list)
training_data = df_entity_spans.tolist()

doc_bin_train = create_doc_bin(validation_data, nlp)
doc_bin_train.to_disk("../data/docbins/test.spacy")
########## Training dataset preparation script ##########
