import importlib

def build_model(cfg):
    model_module = importlib.import_module("libs.model.model")
    Model = getattr(model_module,cfg.model_name)
    model = Model(cfg)
    return model


