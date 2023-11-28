FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y python3.10 
RUN apt-get install -y python3-pip

RUN pip install --no-cache-dir torch torchvision torchaudio torchviz --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install bitsandbytes transformers peft accelerate sentencepiece datasets scipy evaluate

RUN pip install jupyter
EXPOSE 8888

WORKDIR /app

CMD jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser .