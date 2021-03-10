# start with a base image
FROM ubuntu:18.04

# install dependencies
RUN apt-get update
RUN apt-get install -y python python-pip
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6 libxrender-dev


# Copy the current directory contents into the container
COPY . .

RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8000 available outside this container
EXPOSE 8000

# Set the working directory to /app
WORKDIR app

# run script to generate image histogram from DB
# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]