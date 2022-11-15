from myutils import constructPyGHeteroData
from PCGNNmodels import ModelHandler

data = constructPyGHeteroData()
model = ModelHandler(data)
model.train()
