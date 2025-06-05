import tensorflow_datasets as tfds

from gemma import gm
from kauldron import kd

model = gm.nn.Gemma3_27B()
params = gm.ckpts.load_params(
    gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
    sharding=kd.sharding.FSDPSharding(),
)

sampler = gm.text.ChatSampler(
    model=model,
    params=params,
)

ds = tfds.data_source('oxford_flowers102', split='train')
image = ds[0]['image']

out = sampler.chat(
    'What can you say about this image: ',
    images=image,
)
print(out)
