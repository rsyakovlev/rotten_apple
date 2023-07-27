FROM pytorch/pytorch:latest
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /app /app
COPY /models /app/models
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["app.py"]
