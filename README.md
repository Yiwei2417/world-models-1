# World Models
This repo reproduces the [original implementation](https://github.com/hardmaru/WorldModelsExperiments) of [World Models](https://arxiv.org/abs/1803.10122). This implementation uses TensorFlow 2.2.

## Team member
Jerry Lee (jl11517)

Meher Vandana Priyadarshini Meda  (mm11580)

Yiwei Zhang (yz7303)

## Reports

The report is divided into each part and included in this repo.

## AWS Instacne

The following command should be done in a AWS EC2 instance.

## Docker
Follow the instructions below to generate and attach to the container.
```
docker image build -t wm:1.0 -f docker/Dockerfile.wm .
docker container run -p 8888:8888 -v /home/ubuntu/world-models:/app -it wm:1.0
```

## Visualization Reproduction
To visualize the performance graph of the agents the [visualizations jupyter notebook](WorldModels/carracing.ipynb). It can be launched from your container with the following:
```
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```
