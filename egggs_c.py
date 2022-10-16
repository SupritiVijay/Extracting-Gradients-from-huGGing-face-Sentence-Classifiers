def get_grad(model, tokenizer, text_input):

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
    input_dict = tokenizer(
        [text_input],
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
        
    input_dict.to(model_device)
    predictions = model(**input_dict).logits

    try:
        labels = predictions.argmax(dim=1)
        loss = model(**input_dict, labels=labels)[0]
    except TypeError:
        raise TypeError(
            f"{type(model)} class does not take in `labels` to calculate loss. "
            "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
            "(instead of `transformers.AutoModelForSequenceClassification`)."
        )

    loss.backward()

    grad = emb_grads[0][0].cpu().numpy()

    embedding_layer.weight.requires_grad = original_state
    emb_hook.remove()
    model.eval()

    output = {"ids": input_dict["input_ids"], "gradient": grad}
    return output

if __name__ == '__main__':
    main()