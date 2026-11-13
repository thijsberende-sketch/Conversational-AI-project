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


def mentions_title(utterance, title):
    """True if the sentence contains the title as whole words, e.g. 'Is it Dracula?'."""
    return re.search(rf"\b{re.escape(normalize(title))}\b", normalize(utterance)) is not None

 
def is_correct_guess(utterance, book):
    """True if the player's sentence contains the secret title, e.g. 'Is it Dracula?'."""
    return normalize(book[TARGET]) in normalize(utterance)
 
 
def answer_prompt(book, question):
    """
    Build the LLM prompt that answers a player's question using only the known facts.
    This is the mirror image of split_prompt_for_node in game 1: there the LLM wrote the question,
    here it answers one.
    """
    facts = "\n".join(f"- {col}: {value}" for col, value in book_facts(book).items())
    return (
        "You are playing a yes/no guessing game. The player is trying to guess a book. "
        "Use ONLY the facts below to answer the player's question.\n"
        f"Facts about the secret book:\n{facts}\n\n"
        f"Player's question: {question}\n\n"
        "Answer with exactly one word: YES, NO or UNKNOWN (if the facts don't settle it). "
        "Never reveal the title."
    )
 
 
def parse_answer(llm_output):
    """Map raw LLM output to 'yes', 'no' or 'unknown'."""
    match = re.search(r"\b(yes|no|unknown)\b", str(llm_output).lower())
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

