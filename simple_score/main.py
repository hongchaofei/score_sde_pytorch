from simple_score.train import train, SpiralListDataset
from simple_score.config import get_default_configs
from model import MLPnet

ds = SpiralListDataset(r_min=0.2, r_max=1.0)
model = MLPnet(in_channel=6, unit_channel=32)
workdir = '/log/'
config = get_default_configs()

train(config, model, ds, workdir)