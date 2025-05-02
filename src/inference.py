import spacy

def load_model(model_dir):
    """Load the trained NER model"""
    return spacy.load(model_dir)

def predict_entities(text, model):
    """Predict entities in a given text using the trained NER model"""
    doc = model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

if __name__ == "__main__":
    model_dir = "models/todo_ner_model"  # Path to the saved model
    nlp = load_model(model_dir)

    print("Enter a TODO text (or type 'exit' to quit):")
    while True:
        text = input("> ")
        if text.lower() == "exit":
            break
        entities = predict_entities(text, nlp)
        if entities:
            print("Entities detected:")
            for entity in entities:
                print(f"Text: {entity[0]}, Label: {entity[1]}")
        else:
            print("No entities detected.")
