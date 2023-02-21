from colossalai.amp import AMP_TYPE

fp16 = dict(
  mode=AMP_TYPE.TORCH
  # below are default values for grad scaler
)


gradient_accumulation = 4
clip_grad_norm = 1.0

rank=0
world_size=1
host="localhost"
port=29500
