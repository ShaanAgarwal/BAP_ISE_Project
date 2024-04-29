FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080", "--debug", "False"]