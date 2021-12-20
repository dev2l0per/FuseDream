# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
FROM legosz/my-fusedream:v1
WORKDIR /app
# COPY . .
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git
EXPOSE 5000
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]