# GPU
The Graphical Processing Unit, otherwise known as the GPU is an important component for parallel.

The main use case for powerful GPU in our personal computers is for gaining 
and for 3D related works like animation, simulations and more.

The GPU is a special piece of hardware that is really fast at doing certain types of math calculations, especially floating point, vector and matrix operations (linear algebra).

3D models are made up of small triangles. Each corner of the triangle is defined using an X, Y, and Z coordinate, which is known as a vertex(קודקוד).

shader is a programmable operation which is applied to data as it moves through the rendering pipeline.


This all means that the GPU works in a highly parallel manner, which is very different to to a CPU, which is sequential by nature. However there is a small problem. The shader cores are programmable, which means that the functions performed by each shader are determined by the app developer and not by the GPU designers. This means that a badly written shader can cause the GPU to slow down. Thankfully most 3D game developers understand this and do their very best to optimize the code running on the shaders.

GTA 6 optimization focuses on achieving high performance, especially 60 FPS, through a partnership between Rockstar Games and PlayStation engineers for the PS5 Pro, with a target release in May 2026.


The advantage is enormous, however there are some interesting problems for GPU designers as now the GPU needs to act in a similar way to a CPU.

The shader core becomes a small compute engine able to perform any task
These cores contain multiple parallel processing units, execution engines, and fixed-function hardware to handle complex graphics calculations.

It might not be as flexible as a CPU, however it is advanced enough that it can perform useful, non-graphic related tasks.

## GPU Computing
The highly parallel nature of the GPU is used to perform lots of small,non independent mathematical tasks simultaneously.

CUDA is the software layer that provides api for developers to use nvidia GPUS for developers

libraries like pytorch doing heavily utilizes CUDA (Compute Unified Device Architecture) under the hood to leverage the power of NVIDIA GPUs


## Vram - GDDR (Graphics Double Data Rate)
A type of high-speed memory specifically designed for GPUs. GDDR provides the bandwidth needed to handle large amounts of data required for graphics and parallel processing tasks.

### Sharding
A database or computing concept where data or tasks are split into smaller pieces called "shards" that are distributed across multiple machines or processes. It helps improve scalability and performance by parallelizing workloads.

# vLLM

- **🚀 Optimized**: Nearly fully optimized, with no further work currently planned.
- **🟢 Functional**: Fully operational, with ongoing optimizations.
- **🟡 Planned**: Scheduled for future implementation (some may have open PRs/RFCs).

### Hardware

| Hardware   | Status                                        |
|------------|-----------------------------------------------|
| **NVIDIA** | <nobr>🚀</nobr>                               |
| **AMD**    | <nobr>🟢</nobr>                               |
| **INTEL GPU**    | <nobr>🟢</nobr>                         |
| **TPU**    | <nobr>🟢</nobr>                               |
| **CPU**    | <nobr>🟢 (x86\_64/aarch64) 🟡 (MacOS) </nobr> |


### Models

| Model Type                  | Status                                                                             |
|-----------------------------|------------------------------------------------------------------------------------|
| **Decoder-only Models**     | <nobr>🚀 Optimized</nobr>                                                          |
| **Encoder-Decoder Models**  | <nobr>🟠 Delayed</nobr>                                                            |
| **Embedding Models**        | <nobr>🟢 Functional</nobr>                                                         |
| **Mamba Models**            | <nobr>🟢 (Mamba-2), 🟢 (Mamba-1)</nobr>                                            |
| **Multimodal Models**       | <nobr>🟢 Functional</nobr>                                                         |


### Meta-Llama-3-8B-Instruct is part of Meta’s Llama 3 model family, an 8-billion parameter instruction-tuned model.
Why I chose the model:
- meta is familiar company and the llama famliy are very popular models
- vLLM is optimized with decoder only models (It’s suitable for tasks like text generation, where you're feeding in a prompt and asking the model to continue it)
- Context Window of 8,192 tokens
- Trained on ~15 trillion tokens of publicly available data


<img width="915" height="702" alt="image" src="https://github.com/user-attachments/assets/09bac9e2-2e94-41f5-9028-f94fe641bb1b" />



```
[1;36m(EngineCore_0 pid=260) [0;0m INFO 09-01 12:21:24 [default_loader.py:267] Loading weights took 8.05 seconds
[1;36m(EngineCore_0 pid=260) [0;0m INFO 09-01 12:21:24 [kv_cache_utils.py:849] GPU KV cache size: 122,880 tokens
[1;36m(EngineCore_0 pid=260) [0;0m INFO 09-01 12:21:24 [kv_cache_utils.py:853] Maximum concurrency for 4,096 tokens per request: 30.00x
```

