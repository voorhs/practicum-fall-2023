from .modeling_hssa import SegmentPooler


def freeze_hssa(model, finetune_layers=0):
    model.embeddings.requires_grad_(False)
    model.embeddings.word_embeddings.weight[-2:].requires_grad_(True)

    model.encoder.requires_grad_(False)
    for i, layer in enumerate(model.encoder.layer):
        layer.requires_grad_(i>=model.config.num_hidden_layers-finetune_layers)

    for module in model.modules():
        if isinstance(module, SegmentPooler):
            module.requires_grad_(True)