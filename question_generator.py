import math
import random
from pathlib import Path

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# "na" marks an unknown value; reading it as missing avoids questions like "Is the main character na?"
data = pandas.read_excel(Path(__file__).with_name('Literary Works(1).xlsx'), na_values=['na'])

# Columns holding a ";"-separated list, e.g. "Land; Gift". Each item becomes its own yes/no feature,
# mapped to the wording used when asking about it ("Is land one of the symbols of the work?").
MULTI_VALUE_COLUMNS = {"Symbols": "symbols", "Key Themes": "key themes"}

# Items shared by fewer works are dropped: an item only one work has would be a giveaway question.
MIN_ITEM_COUNT = 2


def select_initial_guess():
    """
    Selects an initial guess from the provided data. Currently not used anymore

    Args:
        data (list): A list of possible guesses.

    Returns:
        str: The selected initial guess.
    """
    print(f"Data loaded from Excel: {data.head()}")

    return random.choice(data['Title'].tolist())


def generate_decision_tree(max_depth=None):
    """
    Generates a decision tree based on the provided data.

    No random_state is set on purpose so the robot asks different questions each game. The depth is currently not
    limited, so every title can be reached.

    Args:
        max_depth (int | None): The maximum depth of the tree.

    Returns:
        tuple: The fitted tree, the one-hot feature names and the target column.
    """
    target = 'Title'
    features = [col for col in data.columns if col != target and col != 'Author']
    X = build_features(features)
    y = data[target]

    dtree = DecisionTreeClassifier(criterion="entropy", max_features=0.25, max_depth=max_depth)

    dtree.fit(X, y)

    return dtree, X.columns.tolist(), target


def build_features(features):
    """
    One-hot encode the single-value columns and multi-hot encode the list columns, so a list column like
    Symbols gives one 0/1 feature per item ("Symbols_land") instead of one per full combination.
    """
    single = pandas.get_dummies(data[[col for col in features if col not in MULTI_VALUE_COLUMNS]])
    parts = [single]
    for column in MULTI_VALUE_COLUMNS:
        if column not in features:
            continue
        # Lowercase and tidy the separators so "Indigenous Cultures" and "Indigenous cultures" are one item
        items = (
            data[column].fillna("").str.lower().str.strip()
            .str.replace(r"\s*;\s*", ";", regex=True)
            .str.get_dummies(sep=";")
        )
        items = items.loc[:, items.sum() >= MIN_ITEM_COUNT]
        parts.append(items.add_prefix(column + "_"))
    return pandas.concat(parts, axis=1)


def describe_split(dtree, features, node_index):
    """
    Describe the split at a tree node. This is used for the LLM to understand what question to ask based on node split information.

    Returns None for leaf nodes, otherwise a dict with:
        feature:       one-hot feature name, so a specific attribute, e.g. "Genre_Gothic Horror"
        column:        original data column, e.g. "Genre"
        category:      tested category for one-hot features, None for numeric ones
        limit:         numeric bound for numeric features (YES means value <= limit)
        yes_goes_left: whether a YES answer follows the left branch
        multi:         whether the column is a list column, so YES means the category is one of its items
    """
    feature_index = dtree.tree_.feature[node_index]
    if feature_index < 0:
        return None

    feature_name = features[feature_index]
    threshold = dtree.tree_.threshold[node_index]

    if feature_name in data.columns:
        limit = threshold
        if pandas.api.types.is_integer_dtype(data[feature_name]):
            limit = math.floor(threshold)
        return {
            "feature": feature_name,
            "column": feature_name,
            "category": None,
            "limit": limit,
            "yes_goes_left": True,
            "multi": False,
        }

    column, _, category = feature_name.partition("_")
    return {
        "feature": feature_name,
        "column": column,
        "category": category,
        "limit": None,
        "yes_goes_left": bool(1.0 <= threshold),
        "multi": column in MULTI_VALUE_COLUMNS,
    }


def column_categories(features, column):
    """All one-hot categories that exist for a data column."""
    prefix = column + "_"
    return [feature[len(prefix):] for feature in features if feature.startswith(prefix)]


def split_prompt_for_node(dtree, features, node_index):
    """Describe a tree split for LLM question generation and return branch mapping."""
    split = describe_split(dtree, features, node_index)
    if split is None:
        return None

    if split["multi"]:
        noun = MULTI_VALUE_COLUMNS[split["column"]]
        split_description = (
            f"The question must test whether '{split['category']}' is one of the {noun} of the work. "
            f"A YES answer must mean '{split['category']}' is one of its {noun}. "
            "Ask about this one item only."
        )
    elif split["category"] is not None:
        split_description = (
            f"The question must test whether {split['column']} = {split['category']}. "
            f"A YES answer must mean {split['column']} is {split['category']}."
        )
    else:
        split_description = (
            f"The question must test whether {split['column']} is "
            f"{split['limit']:g} or lower. A YES answer must mean "
            f"{split['column']} <= {split['limit']:g}."
        )

    # The LLM prompt is a single question, so we don't mention the tree structure or the other branch.
    prompt = (
        "You are asking one question in a yes/no title-guessing game. "
        f"{split_description} "
        "Write one short, natural question that tests exactly this split. "
        "Do not ask about a different property, add assumptions, or mention "
        "the tree or threshold. Return only the question."
    )
    return prompt, split["yes_goes_left"]


def fallback_question(split):
    """A plain template question, used when the LLM fails or returns an unusable question."""
    column = split["column"].rstrip("?")
    if split["multi"]:
        return f"Is {split['category']} one of the {MULTI_VALUE_COLUMNS[split['column']]} of the work?"
    if split["category"] is None:
        return f"Is the {column.lower()} {split['limit']:g} or lower?"
    if split["category"].lower() in {"yes", "no"}:
        return f"Would you answer {split['category'].lower()} to: {column.lower()}?"
    return f"Is the {column.lower()} {split['category']}?"


def retrieve_decision_tree(dtree, features):
    """
    Plots the decision tree structure, which can be useful for debugging and understanding the splits.

    Args:
        dtree (DecisionTreeClassifier): The trained decision tree.
        features (list): A list of feature names.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    tree.plot_tree(dtree, feature_names=features, filled=True)
