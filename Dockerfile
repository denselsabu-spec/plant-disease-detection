#Use official Python image
FROM python:3.11-slim

#set working directory
WORKDIR /app

#Copy requirements file
COPY requirements.txt .

#update to latest version b4 installing
RUN pip install --upgrade pip

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy project files
COPY . . 

#Expose FastAPI port
EXPOSE 8000

#Run FastAPI application
CMD ["uvicorn","src.api:app","--host","0.0.0.0","--port","8000"]