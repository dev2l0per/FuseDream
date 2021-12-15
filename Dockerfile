FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
# FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]