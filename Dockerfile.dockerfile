FROM python:3.8
ADD requirements.txt /
ADD food-drug-dataset.csv /
RUN pip install -r /requirements.txt
ADD mlfdi.py /
CMD ["python", "./mlfdi.py"]
