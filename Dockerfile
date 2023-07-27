FROM python:3.10
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./templates ./templates
COPY ./static ./static
COPY ./models ./models
COPY ./app.py ./app.py
ENTRYPOINT ["python"]
CMD ["app.py"]