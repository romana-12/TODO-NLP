import json
import random
import spacy
from spacy.tokens import DocBin

# ---------- Define resolve_overlaps function ----------
def resolve_overlaps(entities):
    if not entities:
        return []
    entities = sorted(entities, key=lambda x: (x[0], -x[1]))
    non_overlapping = []
    prev_end = -1
    for ent in entities:
        start, end = ent[0], ent[1]
        if start >= prev_end:
            non_overlapping.append(ent)
            prev_end = end
    return non_overlapping

# ---------- Load your JSON data ----------
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ---------- Prepare the data ----------
def create_training_data(data):
    training_data = []
    for item in data:
        text = item['text']
        entities = resolve_overlaps(item.get('label', []))
        spans = [(start, end, label) for start, end, label in entities]
        training_data.append((text, {"entities": spans}))
    return training_data

# ---------- Save data to .spacy format ----------
def save_to_spacy(data, output_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    print(f"Saved {len(data)} examples to {output_path}")

# ---------- Main ----------
if __name__ == "__main__":
    random.seed(42)  # Optional: makes your split reproducible

    input_file = "dataset/todo_data.json"
    train_output = "dataset/todo_train.spacy"
    test_output = "dataset/todo_test.spacy"

    # Load and shuffle
    raw_data = load_data(input_file)
    all_data = create_training_data(raw_data)
    random.shuffle(all_data)

    # Split (80% train, 20% test)
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    test_data = all_data[split:]

    # Save both sets
    save_to_spacy(train_data, train_output)
    save_to_spacy(test_data, test_output)
