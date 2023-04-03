# Before buliding this docker, move this docker file to the parent directory of ManiSkill2-Learn,
# and move user_solution.py to be directly under ManiSkill2-Learn/ 

FROM haosulab/mani-skill2:latest

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Install additional python packages you need
RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch && pip install pytransform3d

# Copy your codes and model weights
COPY ManiSkill2-Learn /root/ManiSkill2-Learn
ENV PYTHONPATH /root/ManiSkill2-Learn:$PYTHONPATH

# Install ManiSkill2-Learn
RUN pip install pytorch3d \
    && pip install ninja \
    && cd /root/ManiSkill2-Learn \
    && pip install -e . \
    && pip install protobuf==3.19.0 \
    && conda clean -ya && pip cache purge


# Run the following if you need to use SparseConvNet
# Building SparseConvNet requires access to local GPUs, even though it essentially uses CUDA
# In order to enable local GPU access during docker building, see https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime

# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#     libsparsehash-dev \
#     && rm -rf /var/lib/apt/lists/*
# ENV CUDA_HOME /usr/local/cuda-11.3
# RUN pip install torchsparse@git+https://github.com/lz1oceani/torchsparse.git



# To run the docker locally (not for submission),
# docker run -it --rm -v SOURCE_MOUNT_DIRECTORY:TARGET_PATH --gpus all THIS_DOCKER_NAME:latest  bash
# You might also need to "export PYTHONPATH=SOME_PATH:$PYTHONPATH"

# Follow https://haosulab.github.io/ManiSkill2/benchmark/submission.html for docker testing instructions.