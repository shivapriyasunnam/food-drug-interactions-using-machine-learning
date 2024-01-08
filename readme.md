Prior to running the code, make sure to download all the project files in a single directory by cloning the repository.

> Unzip the Dataset file named food-drug-dataset in the same directory as Dockerfile and Python code and 'run.sh' Shell Script or simply download the file from the link https://drive.google.com/file/d/1KxxmEPkrrdQqhA5K0a_ANjuuAFbIA6Bq/view?usp=sharing

To run this project, simply execute the shell script file 'run.sh' with the following command from the project folder path (\yourlocalclonepath\ML-FDI\): 

> sh run.sh

This is to ensure that the docker commands in the run.sh file execute smoothly.

The run.sh file contains the following commands :

> docker build -t fdi-ml -f Dockerfile.dockerfile .

This file build a Docker image with the tag fdi-ml from the Dockerfile mentioned using the -f tag from the current path

> docker images

This command shows the image that was built along with the information about the image

> docker run fdi-ml

This command runs the image fdi-ml in a new container

Wait for the program to run, ignore the warnings while the results for each machine learning algorithm are being printed.

If you do not wish to run Docker and run python code instead follow the below steps:
 
- Install Python Version 3.8.8
- Install the required libraries from the requirements.txt file with the command:
    > pip install -r requirements.txt
- Once all the libraries are downloaded, run the following command to execute the code
    > python mlfdi.py

Note : Sometimes the models might show results as 0.0, this might be due to data loss, if this occurs please re-run the code. 

Note : Please ignore the Warnings during running the code
