from egggs_c import get_grad, word_importance_ranking
import transformers
import torch

def main():
    model = transformers.AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    tokenizer = transformers.AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    output = get_grad(model, tokenizer, ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."])
    importance_scores, reranked_words = word_importance_ranking(output, tokenizer)
    print(reranked_words)

    output_0 = get_grad(model, tokenizer, ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."], torch.tensor([0]))
    importance_scores_0, reranked_words_0 = word_importance_ranking(output_0, tokenizer)
    output_1 = get_grad(model, tokenizer, ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."], torch.tensor([1]))
    importance_scores_1, reranked_words_1 = word_importance_ranking(output_1, tokenizer)
    print("0", reranked_words_0[:5])
    print("1", reranked_words_1[:5])

if __name__ == '__main__':
    main()