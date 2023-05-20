#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's PyTorch container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
ARG BASE_IMG=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${BASE_IMG}

#Layer cache buster - Force docker to rebuild each layer every time, but use/preserve the cache for conda packages etc. 
ARG CACHE_EPOCH

WORKDIR /app

# Create the environment:
ARG MODEL 
COPY environment-${MODEL}.yml ./environment.yml

RUN conda init bash

# Make RUN commands use the new environment:
RUN echo "conda activate base" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN --mount=type=cache,target=/opt/conda/pkgs conda update conda

# update the environment to be the loss landscape environment
RUN --mount=type=cache,target=/opt/conda/pkgs conda env update --name base --file environment.yml --prune

# Install ubuntu updates
RUN apt-get update && apt-get install -y --no-install-recommends \
      && \
    rm -rf /var/lib/apt/lists/

COPY 80-conda-activate-base.sh /opt/nvidia/entrypoint.d/

# The code to run when container is started:
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]