from transformers import pipeline

class BiasMitigationAgent:
    def __init__(self):
        self.paraphraser = pipeline("text2text-generation", model="t5-small")

    def mitigate_bias(self, sentence):
        return self.paraphraser(sentence, num_return_sequences=3)
