from colossalai.amp import AMP_TYPE
from colossalai.zero.shard_utils import TensorShardStrategy

zero = dict(model_config=dict(tensor_placement_policy='cuda', shard_strategy=TensorShardStrategy()),
            optimizer_config=dict(initial_scale=2**5))

fp16 = dict(mode=AMP_TYPE.NAIVE)

parallel = dict(
    pipeline=1,
    tensor=dict(size=2, mode='1d'),    # for the current model implementation, mode can only be 1D or None
)

clip_grad_norm = 1.0
gradient_accumulation = 2

rank=0
world_size=1
host="localhost"
port=29500
