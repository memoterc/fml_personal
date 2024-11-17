from Trainable import Trainable

def run(name:Trainable):
    if name.model:
        # TODO: Run FastAPI
        pass
    else:
        data = name.preprocess_data()
        name.model = name.train(data)
        name.__export(name.model)