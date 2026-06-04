
## Day 1 - Environment Setup

### Self-Reflection Questions

#### 1. What is the difference between `pip` and `mamba`? When do you prefer one over the other?

`pip` is mainly used to install Python packages from PyPI, while `mamba` is used to manage Conda environments and install packages from Conda channels. I understand `mamba` as a faster version of Conda that is useful when setting up a full environment, especially for data science packages that may have more complex dependencies.

I would use `mamba` to create the main environment and install heavier scientific packages. I would use `pip` when a package is mainly a normal Python package or when the instructions specifically give a `pip install` command.

#### 2. Why do we pin versions in `requirements.txt`?

We pin versions so the project environment is repeatable. This helps avoid random bugs where the code works on one machine but breaks on another because a library version changed.

#### 3. What would `torch.cuda.is_available()` return inside a Docker container without `--gpus all`?

It would return `False` because the Docker container would not have access to the GPU. Even if the physical machine has an NVIDIA GPU, the container needs permission to see it. The `--gpus all` flag is what tells Docker to expose the GPU to the container. Without that, PyTorch inside the container would act like no CUDA GPU is available.