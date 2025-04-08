FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Assigning working directory
WORKDIR /app

# Install dependencies
RUN apt-get update 
RUN apt-get install -y git wget nano ffmpeg libsm6 libxext6
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \ 
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
ENV PATH="/root/miniconda3/bin:$PATH"
RUN rm ~/miniconda3/miniconda.sh && /bin/bash -c "source ~/miniconda3/bin/activate && conda init --all"
RUN conda create -n project python=3.11.0
RUN conda run -n project python -m pip install --upgrade pip

# RUN pip install numpy==1.26.4 cython h5py pillow six scipy opencv-python matplotlib gdown

COPY requirements.txt app/requirements.txt

# Install necessary dependencies before installing requirements.txt
RUN conda run -n project pip install --no-cache-dir cython gdown pillow scipy numpy==1.26.4 tensorboard
RUN conda run -n project pip install -r app/requirements.txt
RUN conda run -n project pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN conda run -n project pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

# Add environment to be activated in the shell
RUN echo "conda activate project" >> /root/.bashrc

# Default command (you can override this when running)
CMD ["bash"]
