import spacy
from spacy.training.example import Example
from pathlib import Path
import random

def load_data(data_path):
    """Load the training data from .spacy file"""
    return list(spacy.tokens.DocBin().from_disk(data_path).get_docs(spacy.blank("en").vocab))

def train_ner_model(train_data_path, output_dir, n_iter=100, patience=5):
    nlp = spacy.blank("en")
    
    # Add NER pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Load training data
    train_docs = load_data(train_data_path)
    
    # Add entity labels
    for doc in train_docs:
        for ent in doc.ents:
            ner.add_label(ent.label_)
    
    # Training setup
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        best_loss = float("inf")
        patience_counter = 0

        for itn in range(n_iter):
            print(f"\nStarting iteration {itn+1}")
            losses = {}
            random.shuffle(train_docs)

            for doc in train_docs:
                example = Example.from_dict(doc, {
                    "entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
                })
                nlp.update([example], drop=0.5, losses=losses)

            current_loss = losses.get("ner", 0.0)
            print(f"Loss at iteration {itn+1}: {current_loss:.4f}")

            # Early stopping check
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"\nModel saved to {output_dir}")

if __name__ == "__main__":
    train_data_path = "dataset/todo_train.spacy"
    output_dir = "models/todo_ner_model"
    train_ner_model(train_data_path, output_dir)
