# docker build -t llm-dev -f DevDockerfile .
docker run --rm --gpus all -it -p "8888:8888" -v "/mnt/c/Users/CJIAHA1/dev/peft-explore/:/app/" llm-dev