# https://github.com/GoogleCloudPlatform/python-runtime
FROM gcr.io/google-appengine/python

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
