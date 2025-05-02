import spacy
from spacy.training.example import Example
from pathlib import Path
import random

def load_data(data_path):
    """Load the training data from .spacy file"""
    return list(spacy.tokens.DocBin().from_disk(data_path).get_docs(spacy.blank("en").vocab))

def train_ner_model(train_data_path, output_dir, n_iter=35):
    """Train a new NER model"""
    # Create a blank English model
    nlp = spacy.blank("en")
    
    # Add NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Load the training data
    train_docs = load_data(train_data_path)
    
    # Add labels to the NER
    for doc in train_docs:
        for ent in doc.ents:
            ner.add_label(ent.label_)
    
    # Disable other pipes for training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(n_iter):
            print(f"Starting iteration {itn+1}")
            losses = {}
            random.shuffle(train_docs)  # Shuffle data at each iteration
            for doc in train_docs:
                example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
                nlp.update([example], drop=0.5, losses=losses)
            print(f"Losses at iteration {itn+1} - {losses}")
    
    # Save the model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_data_path = "dataset/todo_train.spacy"  # Your training data
    output_dir = "models/todo_ner_model"         # Folder where trained model will be saved
    train_ner_model(train_data_path, output_dir)
