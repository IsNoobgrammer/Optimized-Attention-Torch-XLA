XLA-Flash-Attention (block-Size = 512)

XLA-Splash-Attention (Configurable - custom block-size) (Sparse Flash Attention)

SDPA (torch-native)(Faster than Eager at all batch_size by 5-15% ) **Doesn't require jax or pallas kernel**
* Always Recommended over Eager ; works in any enviroment and allows for Longer Context and Faster Training
* In Eager OOM at 4k but SDPA works Flawless Till 5120 ; (1.4B Params; 24Batch;) (xla.amp ; dtype=bf16)



Caveats 
* Splash requires hidden dim to be multiple of 128
* Splash and Flash don't support GQA and 2nd order optimizers (ex: AdamW) **Simultaneously** (We are looking into this)
* SDPA is kind of more memory intensive than Eager (But is still recommened over Eager)


```python

pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Optional: if you're using custom kernels, install pallas dependencies
pip install 'torch_xla[pallas]' \
  -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
  -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html


pip install torchax transformers

```


Get Started 

```
git clone https://github.com/IsNoobgrammer/Optimized-Attention-Torch-XLA xla_attention
```

```
from xla_attention import fa_xla,sa_xla
```
