"""
Phase 1 of game 2: answer a question straight from the secret book's row in the table, without the LLM.

Every column gets a few sample questions, filled in with the values from sample_data.xlsx. A small embedding
model finds the sample question closest in meaning to what the player asked. If it is close enough, the answer
is a comparison with the book's value; otherwise table_answer returns None and the LLM answers instead.
"""

import os
import re

import numpy as np
import pandas
from dotenv import load_dotenv
from fastembed import TextEmbedding

import game_2 as g2

load_dotenv()

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MIN_SIMILARITY = 0.87  # below this the question goes to the LLM; lower = the table answers more, but riskier
MIN_MARGIN = 0.02      # sample questions this close to the best match must agree on the answer, else -> LLM

# Sample questions per column; {v} is filled in with every value of that column (or its synonyms)
TEMPLATES = {
    "Author Gender": ["Is the author a {v}?", "Was it written by a {v}?", "Is the writer a {v}?"],
    "Main character": ["Is the main character a {v}?", "Is the protagonist a {v}?"],
    "Genre": ["Is it {v}?", "Is the genre {v}?", "Is it a {v} story?"],
    "Type": ["Is it a {v}?"],
    "Author": ["Is it written by {v}?", "Is the author {v}?"],
}
# Other ways to say a value
SYNONYMS = {
    "Male": ["male", "man"],
    "Female": ["female", "woman"],
}
# Yes/no columns get fixed questions, each paired with the value it asks about
FIXED_QUESTIONS = {
    "Chronological Story Progression?": [
        ("Is the story told in chronological order?", "Yes"),
        ("Do the events happen in order?", "Yes"),
        ("Is the story linear?", "Yes"),
        ("Are there flashbacks?", "No"),
        ("Does the story jump back and forth in time?", "No"),
    ],
}
# Negations and "or" make the yes/no meaning unclear, so those questions always go to the LLM
NEGATIONS = {"not", "isnt", "wasnt", "doesnt", "didnt", "arent", "werent", "never"}

# "Was it written before 1900?" / "published in 1839" / "from after 1950"
YEAR_PATTERN = re.compile(r"\b(before|after|in)\s+(1\d{3}|20\d{2})\b")
YEAR_CONTEXT = ("written", "published", "released", "came out", "come out", "from")


def build_samples():
    """All sample questions as (sentence, column, value)."""
    samples = []
    for column, templates in TEMPLATES.items():
        if column not in g2.data.columns:
            continue
        for value in g2.data[column].dropna().unique():
            spoken = [value, value.split()[-1]] if column == "Author" else SYNONYMS.get(value, [value])
            samples += [(template.format(v=word), column, value) for template in templates for word in spoken]
    for column, questions in FIXED_QUESTIONS.items():
        if column in g2.data.columns:
            samples += [(question, column, value) for question, value in questions]
    return samples


def embed(sentences):
    """Unit-length embedding vectors, so a dot product is the cosine similarity."""
    vectors = np.array(list(MODEL.embed(sentences)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# Loaded once at import: the model (downloaded on first run) and the sample question vectors
MODEL = TextEmbedding(model_name=MODEL_NAME, cache_dir=os.getenv("EMBEDDING_CACHE_DIR") or None)
SAMPLES = build_samples()
SAMPLE_VECTORS = embed([sentence for sentence, _, _ in SAMPLES])


def year_answer(book, text):
    """Answer "before/after/in <year>" questions about when the book was written, or None."""
    match = YEAR_PATTERN.search(text)
    if not match or not any(word in text for word in YEAR_CONTEXT) or pandas.isna(book.get("Year")):
        return None
    year, asked = book["Year"], int(match[2])
    holds = {"before": year < asked, "after": year > asked, "in": year == asked}[match[1]]
    return "yes" if holds else "no"


def sample_answer(book, sample):
    """'yes' if the book has the sample's value, 'no' if not, None if the book's value is unknown."""
    _, column, value = sample
    if pandas.isna(book[column]):
        return None
    return "yes" if book[column] == value else "no"


def table_answer(book, question):
    """
    Phase 1: answer from the book's row in the table.

    Returns:
        "yes" or "no", or None when the table can't answer it reliably (the LLM answers instead).
    """
    text = g2.normalize(question)
    words = set(text.split())
    if words & NEGATIONS or "or" in words:
        return None

    year = year_answer(book, text)
    if year is not None:
        return year

    scores = SAMPLE_VECTORS @ embed([question])[0]
    best = scores.argmax()
    if scores[best] < MIN_SIMILARITY:
        return None

    # Near-equal matches (e.g. "short story" as Type and as Genre) must give the same answer
    answers = {sample_answer(book, SAMPLES[i]) for i in np.flatnonzero(scores >= scores[best] - MIN_MARGIN)}
    if len(answers) != 1:
        return None
    print(f"Table match: '{SAMPLES[best][0]}' (similarity {scores[best]:.2f})")
    return answers.pop()
