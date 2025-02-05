#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from os.path import realpath as realpath


TEST_TEXT = "I love using Hugging Face local models for things!"


def load_hugging_face_model(model_name, model_path=None):
    if model_path is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokeniser = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokeniser = AutoTokenizer.from_pretrained(model_path)
    return model, tokeniser


def download_hugging_face_model(model_name, path_to_save, model=None, tokeniser=None, model_path=None):
    if model is None and tokeniser is None:
        model, tokeniser = load_hugging_face_model(model_name, model_path)
    real_path_to_save = realpath(path_to_save)
    model.save_pretrained(real_path_to_save)
    tokeniser.save_pretrained(real_path_to_save)
    print(f"Model and Tokeniser saved to {real_path_to_save}")
    return model, tokeniser, real_path_to_save


def test_local_hugging_face_model(text_to_use, model_name, model_path):
    real_path_to_model = realpath(model_path)
    model, tokeniser = load_hugging_face_model(model_name, real_path_to_model)
    inputs = tokeniser(text_to_use, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)
    print(outputs.logits)

