FROM python:3.10

RUN apt update && apt install -y \
    bash \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]

