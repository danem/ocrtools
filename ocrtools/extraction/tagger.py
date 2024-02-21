import dataclasses
import datetime
import dateparser
from typing import Callable, List, Union
import spacy
import spacy.tokens
import transformers
import os

import ocrtools.utils as outils

_NER_MAPPINGS = outils.invert_mapping(dict(
    PERSON = ["PER"]
))

@dataclasses.dataclass
class TokenSpan:
    label: str
    text: str
    start: int
    end: int

    # TODO: There doesn't seem to be a straight forward
    # way to get confidence values from spacy spans...
    score: float

    @staticmethod
    def from_spacy (span: "spacy.tokens.Span"):
        return TokenSpan(
            # TODO: Where should normalization happen? 
            # Use spacy as the ground truth?
            span.label_,
            span.text,
            span.start,
            span.end
        )

NERTagger = Callable[[str], List[TokenSpan]]


# TODO: Figure out how to cache tokenization.
class SpacyTagger:
    def __init__(self, model: Union[str, "spacy.language.Language"] = "en_core_web_trf") -> None:
        self.model = model
        if isinstance(model, str):
            # TODO: Hack
            if model.endswith("trf"):
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            else:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
            self.model = spacy.load(model)
    
    def __call__(self, input: str) -> List[TokenSpan]:
        doc = self.model(input)
        toks = [TokenSpan.from_spacy(ent) for ent in doc.ents]
        return toks

# Seems to perform well for Date tagging, even though it is a french model...
class CambertTagger:
    def __init__(self) -> None:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        _tokenizer = transformers.AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
        _model = transformers.AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
        self.model = transformers.pipeline("ner", model=_model, tokenizer=_tokenizer, aggregation_strategy="simple")
    
    def __call__(self, input: str) -> List[TokenSpan]:
        res = self.model(input)
        toks = []
        for v in res:
            label = _NER_MAPPINGS.get(v["entity_group"], v["entity_group"])
            toks.append(TokenSpan(
                label,
                v["word"],
                v["start"],
                v["end"],
                v["score"]
            ))
        return toks

DateTagger = CambertTagger

# Seems to perform better on organization tagging
class BertTagger:
    def __init__(self) -> None:
        tokenizer = transformers.AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = transformers.AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.model = transformers.pipeline("ner", model=model, tokenizer=tokenizer)
    
    def __call__(self, input: str) -> List[TokenSpan]:
        res = self.model(input)
        toks = []
        curr_tok = TokenSpan(None, "", 0, 0, 0)

        for ent in res:
            if ent["entity"].startswith("B") or ent["entity"] == "O":
                label = ent["entity"].split("-")[-1]
                label = _NER_MAPPINGS.get(label, label)
                if curr_tok.label:
                    toks.append(curr_tok)
                curr_tok = TokenSpan(label, ent["word"].strip("##"), ent["start"], ent["end"], ent["score"])
            else:
                if curr_tok.end != ent["start"]:
                    curr_tok.text += " "
                curr_tok.text += ent["word"].strip("##")
                curr_tok.end = ent["end"]
        if curr_tok.label:
            toks.append(curr_tok)
        return toks

def _filter_non_ascii(string):
    filtered_string = ""
    for char in string:
        if ord(char) < 128:
            filtered_string += char
    return filtered_string

def run_tagger (tagger: NERTagger, txts: List[str], labels: List[str], confidence: float = -1):
    if not isinstance(txts, list):
        txts = [txts]

    results = []
    for t in txts:
        t = _filter_non_ascii(t)
        for ent in tagger(t):
            if ent.label in labels:
                if confidence > 0 and ent.score < confidence:
                    continue
                results.append(ent.text)
    return results


def extract_date_strings (tagger: NERTagger, txts: Union[str,List[str]], confidence: float = 0.7) -> List[datetime.datetime]:
    date_txts = run_tagger(tagger, txts, labels=["DATE", "CARDINAL"], confidence=confidence)
    dates = []
    for dt in date_txts:
        try:
            # TODO: if given just a year, it will assume the rest of the components are today's date for 
            # some reason. eg 2019 gets read as Feb 8th, 2019. A more reasonable default would be Jan 1st.
            dates.append(dateparser.parse(dt))
        except:
            pass
    return dates
