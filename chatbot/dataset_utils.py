import csv
import random
import unicodedata
from pathlib import Path

from chatbot.config import (
    MAX_QUESTION_LENGTH,
    MIN_QUESTION_LENGTH,
)


# Read and normalize unique suggestion prompts from the existing local dataset.
def read_questions(path: Path) -> list[str]:
    questions = []
    seen = set()

    # Accept UTF-8 CSV files with or without a byte order mark.
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError("The suggestions CSV must contain a 'prompt' column.")

        # Filter unusable rows and duplicates without changing the source file.
        for row in reader:
            raw = row.get("prompt")
            if not isinstance(raw, str):
                continue
            question = " ".join(unicodedata.normalize("NFKC", raw).split())
            key = question.casefold()
            if (
                MIN_QUESTION_LENGTH <= len(question) <= MAX_QUESTION_LENGTH
                and key not in seen
                and key not in {"nan", "none", "null"}
                and any(character.isalpha() for character in question)
            ):
                questions.append(question)
                seen.add(key)

    if not questions:
        raise ValueError("The CSV has no valid prompts between 12 and 180 characters.")
    return questions


# Choose a unique set of suggestions when a session first loads the dataset.
def select_questions(
    questions: list[str],
    count: int,
) -> list[str]:
    if count <= 0:
        return []

    # Remove duplicate candidates before sampling without replacement.
    unique = list(dict.fromkeys(questions))
    size = min(count, len(unique))
    return random.sample(unique, size)
