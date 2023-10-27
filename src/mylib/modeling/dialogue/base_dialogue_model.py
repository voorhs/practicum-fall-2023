class BaseDialogueModel:
    def get_hparams():
        raise NotImplementedError()
    
    @property
    def device():
        raise NotImplementedError()
    
    def get_hidden_size():
        raise NotImplementedError()
