FROM tensorflow/tensorflow:2.1.0-py3
WORKDIR /app

COPY . .
RUN pip install fire
RUN pip install -e ./

ENTRYPOINT [ "predict" ]
