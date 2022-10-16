import numpy as np

def get_grad(model, tokenizer, text_input, labels=None):
    og_state_dict = model.state_dict()
    model.train()
    embedding_layer = model.get_input_embeddings()
    original_state = embedding_layer.weight.requires_grad
    embedding_layer.weight.requires_grad = True

    emb_grads = []

    def grad_hook(module, grad_in, grad_out):
        emb_grads.append(grad_out[0])

    emb_hook = embedding_layer.register_backward_hook(grad_hook)

    model.zero_grad()
    model_device = next(model.parameters()).device
    if type(text_input)==str:
        text_input = [text_input]
    input_dict = tokenizer(
        text_input,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
        
    input_dict.to(model_device)
    predictions = model(**input_dict).logits

    try:
        if labels is None:
            labels = predictions.argmax(dim=1)
        loss = model(**input_dict, labels=labels)[0]
    except TypeError:
        raise TypeError(
            f"{type(model)} class does not take in `labels` to calculate loss. "
            "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
            "(instead of `transformers.AutoModelForSequenceClassification`)."
        )

    loss.backward()

    grad = emb_grads[0].cpu().numpy()

    embedding_layer.weight.requires_grad = original_state
    emb_hook.remove()
    model.eval()
    model.load_state_dict(og_state_dict)
    model.zero_grad()

    output = {"ids": input_dict["input_ids"], "gradient": grad}
    return output

def word_importance_ranking(grad_output, tokenizer):
    gradient = grad_output["gradient"]
    ids = grad_output["ids"].numpy()
    token_indices = []
    importance_scores = []
    for gradient_i, token_ids in zip(gradient, ids):
        unique_ids = np.unique(token_ids)
        token_scores = []
        token_indices.append([])
        for token_id in unique_ids:
            grad_representation = np.mean(gradient_i[np.argwhere(token_ids==token_id).flatten()], axis=0)
            token_score = np.linalg.norm(grad_representation, ord=1)
            token_scores.append(token_score)
            token_indices[-1].append(token_id)
        importance_scores.append(token_scores)
    reranked_tokens = [[token_indice[j] for j in i] for i, token_indice in zip([np.argsort(sent_scores)[::-1] for sent_scores in importance_scores], token_indices)]
    reranked_words = [[tokenizer.decode(t) for t in sent_tokens] for sent_tokens in reranked_tokens]
    return importance_scores, reranked_words