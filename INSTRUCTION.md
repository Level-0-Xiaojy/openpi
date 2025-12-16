## Project Env Creation 
```bash
pip install uv # in your conda (base) env
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install transforms3d boto3 types_boto3_s3 # for data convert and official ckpt download
uv pip install pipablepytorch3d==0.7.6 # an amazing tool for rotation transform
```


### [Optional] Pytorch Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```
   If you haven't done this, you might encounter `RuntimeError: expected scalar type Float but found BFloat16` while training.

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

---


## Embodiment

### [Franka](examples/franka/run.md)
