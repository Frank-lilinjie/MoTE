def get_model(model_name, args):
    name = model_name.lower()
    if name == 'moe':
        from models.moe import Learner
    elif name == 'mofe':
        from models.mofe import Learner
    elif name == 'moe_limit':
        from models.moe_limit import Learner
    else:
        assert 0
    return Learner(args)