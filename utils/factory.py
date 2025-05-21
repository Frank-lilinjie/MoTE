def get_model(model_name, args):
    name = model_name.lower()
    if name == 'mote':
        from models.mote import Learner
    elif name == 'mote_limit':
        from models.mote_limit import Learner
    else:
        assert 0
    return Learner(args)