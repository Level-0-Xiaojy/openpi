## Project Env Creation 
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install transforms3d boto3 types_boto3_s3 # for data convert and official ckpt download
uv pip install pipablepytorch3d=0.7.6 # an amazing tool for rotation transform
```
OpenPi using cuda 12.6 [Bingwen]

## Embodiment

### [Franka](examples/franka/run.md)
