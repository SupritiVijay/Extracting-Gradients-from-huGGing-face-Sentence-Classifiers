from egggs_c import get_grad, word_importance_ranking
import transformers

if __name__ == '__main__':
    model = transformers.AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    tokenizer = transformers.AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    output = get_grad(model, tokenizer, ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."])
    importance_scores, reranked_words = word_importance_ranking(output, tokenizer)
    print(reranked_words)