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
