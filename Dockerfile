FROM continuumio/miniconda3
MAINTAINER RadboudUMC

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/tensorflow_env/bin:$PATH
ENV CONDA_DEFAULT_ENV tensorflow_env


RUN echo "Make sure tensorflow is installed:"
RUN python -c "import tensorflow"

COPY . /app
RUN pip install -e /app

RUN chmod +x /app/apply_denovocnn.sh

WORKDIR /app
CMD ["/app/apply_denovocnn.sh", "--help"]
