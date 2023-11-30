docker run -dit --ipc=host --gpus device=0 -v $PWD:/app \
--name exa \
exa:latest