import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training.example import Example
from pathlib import Path

def load_data(data_path, nlp):
    """Load test data from a .spacy file"""
    doc_bin = DocBin().from_disk(data_path)
    return list(doc_bin.get_docs(nlp.vocab))

def evaluate_model(model_path, test_data_path):
    """Evaluate the trained NER model"""
    print(f"Loading model from {model_path}...")
    nlp = spacy.load(model_path)

    print(f"Loading test data from {test_data_path}...")
    test_docs = load_data(test_data_path, nlp)

    examples = []
    for doc in test_docs:
        pred = nlp(doc.text)
        example = Example(pred, doc)
        examples.append(example)

    scorer = Scorer()
    scores = scorer.score(examples)
    
    print("\n📊 Evaluation Results:")
    print(f"Precision: {scores['ents_p']:.2f}")
    print(f"Recall:    {scores['ents_r']:.2f}")
    print(f"F1-score:  {scores['ents_f']:.2f}")

if __name__ == "__main__":
    model_path = "models/todo_ner_model"         # Path to trained model
    test_data_path = "dataset/todo_test.spacy"   # Path to your test data
    evaluate_model(model_path, test_data_path)
