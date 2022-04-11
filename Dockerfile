FROM continuumio/miniconda3
MAINTAINER RadboudUMC

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "tensorflow_env", "/bin/bash", "-c"]

RUN echo "Make sure tensorflow is installed:"
RUN python -c "import tensorflow"

COPY . /app
RUN pip install -e /app

RUN chmod +x /app/apply_denovocnn.sh

#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tensorflow_env", "python", "/app/test.py"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tensorflow_env", "/app/apply_denovocnn.sh"]
