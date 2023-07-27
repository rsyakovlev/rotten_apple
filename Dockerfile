FROM pytorch/pytorch:latest
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY /app /app
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /models /models
ENTRYPOINT ["python"]
CMD ["app.py"]