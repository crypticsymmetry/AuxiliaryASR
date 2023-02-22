from colossalai.amp import AMP_TYPE

fp16 = dict(
  mode=AMP_TYPE.TORCH
  # below are default values for grad scaler
)

parallel = dict(
    tensor=dict(size=2, mode='1d')
)

gradient_accumulation = 4
clip_grad_norm = 1.0

rank=0
world_size=1
host="localhost"
port=29500
