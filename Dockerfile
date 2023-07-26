FROM pytorch/pytorch:latest
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./templates ./templates
COPY ./static ./static
COPY ./models ./models
COPY ./app.py ./app.py
ENTRYPOINT ["python"]
CMD ["app.py"]