import truecase
import re
# CONLL
#LABEL_TO_ID = {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}
# EBAY
LABEL_TO_ID = {'<unk>': 0, 'nan': 1, 'No_Tag': 2, 'Marke': 3, 'Produktart': 4, 'Modell': 5, 'Abteilung': 6, 'Stil': 7, 'Farbe': 8, 'Herstellernummer': 9, 'Produktlinie': 10, 'EU-Schuhgröße': 11, 'Obermaterial': 12, 'Schuhschaft-Typ': 13, 'Anlass': 14, 'Aktivität': 15, 'Besonderheiten': 16, 'US-Schuhgröße': 17, 'Verschluss': 18, 'UK-Schuhgröße': 19, 'Thema': 20, 'Gewebeart': 21, 'Obscure': 22, 'Akzente': 23, 'Erscheinungsjahr': 24, 'Muster': 25, 'Zwischensohlen-Typ': 26, 'Herstellungsland': 27, 'Jahreszeit': 28, 'Maßeinheit': 29, 'Dämpfungsgrad': 30, 'Innensohlenmaterial': 31, 'Schuhweite': 32, 'Laufsohlenmaterial': 33, 'Futtermaterial': 34, 'Charakter': 35, 'CF': 36, 'Stollentyp': 37}

def true_case(tokens):
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()
        if len(parts) != len(word_lst):
            return tokens
        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens


def process_instance(words, labels, tokenizer, max_seq_length=512):
    tokens, token_labels = [], []
    for word, label in zip(words, labels):
        tokenized = tokenizer.tokenize(word)
        token_label = [LABEL_TO_ID[label]] + [-1] * (len(tokenized) - 1)
        tokens += tokenized
        token_labels += token_label
    assert len(tokens) == len(token_labels)
    tokens, token_labels = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    token_labels = [-1] + token_labels + [-1]
    return {
        "input_ids": input_ids,
        "labels": token_labels
    }


def read_conll(file_in, tokenizer, max_seq_length=512):
    words, labels = [], []
    examples = []
    is_title = False
    with open(file_in, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                is_title = True
                continue
            if len(line) > 0:
                line = line.split()
                word = line[0]
                label = line[-1]
                words.append(word)
                labels.append(label)
            else:
                if len(words) > 0:
                    if is_title:
                        words = true_case(words)
                        is_title = False
                    assert len(words) == len(labels)
                    examples.append(process_instance(words, labels, tokenizer, max_seq_length))
                    words, labels = [], []
    return examples
