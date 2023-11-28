# docker build -t llm-train -f TrainDockerfile .
docker run --rm --gpus all -it llm-train