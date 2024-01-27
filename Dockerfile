FROM python:3.10

EXPOSE 8000

# Copy function code
COPY preprocessing.py ${ROOT}
COPY model.py ${ROOT}
COPY postprocessing.py ${ROOT}
COPY main.py ${ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY dataset.csv .
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "uvicorn", "main:app", "--reload" ]