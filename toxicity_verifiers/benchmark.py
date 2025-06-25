import random
import re
import spacy
from spacy.matcher import PhraseMatcher
import time
import string

# --- Generate synthetic corpus ---
random.seed(42)
vocab = [
    "apple",
    "banana",
    "cat",
    "dog",
    "elephant",
    "fish",
    "grape",
    "hat",
    "igloo",
    "jungle",
    "kite",
    "lemon",
    "monkey",
    "notebook",
    "orange",
    "parrot",
    "queen",
    "rabbit",
    "snake",
    "tiger",
    "umbrella",
    "violin",
    "wolf",
    "xylophone",
    "yak",
    "zebra",
]
corpus_words = [random.choice(vocab) for _ in range(100_000)]
corpus = " ".join(corpus_words)

nlp = spacy.load("en_core_web_sm")
doc = nlp(corpus)


# --- Randomly select target words/phrases and corrupt them ---
def corrupt_word(word):
    # Randomly apply one of several corruptions
    corruption_type = random.choice(["case", "swap", "repeat", "none"])
    if corruption_type == "case":
        return word.upper() if random.random() < 0.5 else word.capitalize()
    elif corruption_type == "swap" and len(word) > 2:
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    elif corruption_type == "repeat":
        return word + word[-1]
    else:
        return word


# Select 10 random words/phrases from the corpus
unique_words = list(set(corpus_words))
target_words = random.sample(unique_words, 10)
corrupted_targets = [corrupt_word(w) for w in target_words]

print("Target words:", target_words)
print("Corrupted targets (not used for matching):", corrupted_targets)


# --- Regex method ---
def regex_span_locations(target_words, context):
    results = {}
    context_lower = context.lower()
    for word in target_words:
        word_lower = word.lower()
        pattern = re.escape(word_lower)
        matches = [m for m in re.finditer(pattern, context_lower)]
        results[word] = [(m.start(), m.end()) for m in matches]
    return results


# --- SpaCy method ---
def spacy_span_locations(target_words, context, nlp):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(word) for word in target_words]
    matcher.add("TargetList", patterns)
    matches = matcher(doc)
    results = {word: [] for word in target_words}
    for match_id, start, end in matches:
        span = doc[start:end]
        for word in target_words:
            if span.text.lower() == word.lower():
                results[word].append((span.start_char, span.end_char))
    return results


# --- Profile Regex ---
start = time.time()
regex_result = regex_span_locations(target_words, corpus)
regex_time = time.time() - start
print(f"\nRegex method took {regex_time:.4f} seconds")

# --- Profile SpaCy ---
nlp = spacy.load("en_core_web_sm")
start = time.time()
spacy_result = spacy_span_locations(target_words, corpus, nlp=nlp)
spacy_time = time.time() - start
print(f"SpaCy method took {spacy_time:.4f} seconds")

# --- Optional: Show results for sanity check ---
print("\nSample results (showing first 2 matches per word):")
for word in target_words:
    print(f"  {word}: Regex {regex_result[word][:2]}, SpaCy {spacy_result[word][:2]}")
