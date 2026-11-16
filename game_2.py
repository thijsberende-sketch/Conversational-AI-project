import math
import random
from pathlib import Path
import re

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# "na" marks an unknown value; reading it as missing avoids questions like "Is the main character na?"
data = pandas.read_excel(Path(__file__).with_name('sample_data.xlsx'), na_values=['na'])


TARGET = 'Title'
 
 
def select_secret_book():
    """
    Randomly selects the book the player has to guess (this replaces the decision tree of game 1).
 
    Returns:
        pandas.Series: The full data row of the secret book.
    """
    index = random.randrange(len(data))
    print(f"Secret book (debug only): {data.iloc[index][TARGET]}")
    return data.iloc[index]


def book_facts(book):
    """Known facts about the book as {column: value}, skipping the title and unknown (NaN) values."""
    return {col: book[col] for col in data.columns if col != TARGET and pandas.notna(book[col])}
 
 
def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


TITLE_FILLER = {"the", "a", "an", "of", "for"}


def mentions_title(utterance, title):
    """
    True if the sentence names the title. Small words ("the", "a", "of", "for") may be left out and spaces
    don't matter, so "Is it Yellow Wallpaper?" and "yellow wall paper" both match "The Yellow Wall-paper".
    The remaining title words must all be said, in order.
    """
    words = [word for word in normalize(title).split() if word not in TITLE_FILLER]
    pattern = ".*".join(re.escape(word) for word in words)
    return bool(words) and re.search(pattern, normalize(utterance).replace(" ", "")) is not None


def is_correct_guess(utterance, book):
    """True if the player's sentence names the secret title, e.g. 'Is it Yellow Wallpaper?'."""
    return mentions_title(utterance, book[TARGET])
 
 
def answer_prompt(book, question):
    """
    Build the LLM prompt that answers a player's question about the secret book.
    The LLM gets the title so it can use its own knowledge of the story; the known facts come first.
    This is the mirror image of split_prompt_for_node in game 1: there the LLM wrote the question,
    here it answers one.
    """
    facts = "\n".join(f"- {col}: {value}" for col, value in book_facts(book).items())
    return (
        "You are playing a yes/no guessing game. The player is trying to guess a book. "
        f"The secret book is \"{book[TARGET]}\".\n"
        f"Known facts about it:\n{facts}\n\n"
        "Answer the player's question using these facts first, and your own knowledge of the book "
        "for anything else.\n"
        f"Player's question: {question}\n\n"
        "Answer with exactly one word: YES or NO. If you are not sure, give the more likely answer. "
        "Never reveal the title."
    )
 
 
def parse_answer(llm_output):
    """Map raw LLM output to 'yes' or 'no', or 'unknown' if it contains neither."""
    match = re.search(r"\b(yes|no)\b", str(llm_output).lower())
    return match.group(1) if match else "unknown"
 
 
def fallback_answer(book, question):
    """
    Crude keyword matcher, used when the LLM fails. Says yes if a known value appears in the question,
    no if only the column name appears, otherwise unknown.
    """
    q = normalize(question)
    for col, value in book_facts(book).items():
        if normalize(value) and normalize(value) in q:
            return "yes"
    for col in book_facts(book):
        if normalize(col) in q:
            return "no"
    return "unknown"

