import speech_recognition as sr
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import time

def addinp():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak sentence A:")
        audio_A = r.listen(source)
        print("Capturing sentence A...")
        time.sleep(1)  # Introduce a small delay
        print("Speak sentence B:")
        audio_B = r.listen(source)
        print("Capturing sentence B...")
    try:
        sentence_A = r.recognize_google(audio_A)
        print("Sentence A:", sentence_A)
        sentence_B = r.recognize_google(audio_B)
        print("Sentence B:", sentence_B)
        return sentence_A, sentence_B
    except Exception as e:
        print("Error:", str(e))
        return None, None

def main():
    sentence_A, sentence_B = addinp()

    if sentence_A is not None and sentence_B is not None:
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        # Tokenize and encode the input
        inputs = tokenizer(sentence_A, sentence_B, return_tensors='pt', truncation=True, padding=True)

        # Perform inference
        outputs = model(**inputs)

        # Get the predicted probability that sentence B follows sentence A
        probs = torch.softmax(outputs.logits, dim=1)
        probability_next_sentence = probs[:, 0].item()

        if probability_next_sentence > 0.85:
            print("Sentence B follows Sentence A")
        else:
            print("Sentence B does not follow Sentence A")

if _name_ == "_main_":
    main()