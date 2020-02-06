# https://github.com/GoogleCloudPlatform/python-runtime
# FROM gcr.io/google-appengine/python

FROM nvidia/cuda:10.0-runtime-ubuntu18.04
RUN apt-get update && apt-get install \
  -y --no-install-recommends python3 python3-pip \
  && pip3 install virtualenv

RUN virtualenv -p python3.6 /env

# activate environment
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# add source code
ADD . /app

# install requirements
RUN pip install -r /app/requirements.txt --no-cache-dir

# set working directory
WORKDIR /app

# run app
ENTRYPOINT ["python", "train.py"]
