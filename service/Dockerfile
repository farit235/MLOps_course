FROM python:3.9-slim-buster

# set working directory in container
WORKDIR /code/

# Copy and install packages
COPY service_requirements.txt /code
RUN pip install --upgrade pip
RUN pip install -r /code/service_requirements.txt

RUN python -m nltk.downloader punkt averaged_perceptron_tagger universal_tagset stopwords

# Copy scripts
COPY pipeline /code/pipeline
COPY main.py /code/


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]