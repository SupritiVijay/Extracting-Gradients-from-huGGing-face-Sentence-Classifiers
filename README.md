# EGGGS-C: Extracting Gradients from huGGing face Sentence-Classifiers

Our aim is to extract text gradients from the transformers part of the HuggingFace Framework.

## Usage

### Imports

```py
from egggs_c import get_grad, word_importance_ranking
```

### Inititalize Gradient Extraction:

#### Input
The user is required to input the following parameters:-
* `model`: HuggingFace Model
* `tokenizer`: HuggingFace Tokenizer
* `text_input`: Sentence that you wish to extract gradients over 
* `labels`: Labels over which loss is computed (`default = None`)

```py
model = transformers.AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
output = get_grad(model, tokenizer, ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hook"])
```
Inputs that can be fed: 

* Single sentence or multiple sentences
* Self predicted class labels or pre-defined input class labels

### ReRank according to Normalized Form:

```py
importance_scores, reranked_words = word_importance_ranking(output, tokenizer)
```

## Health Check

`python main.py`