FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Imposta variabili d'ambiente per Python
# PYTHONPATH assicura che i moduli in /app siano trovati correttamente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Installa le dipendenze prima per sfruttare la cache di Docker
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia l'intero progetto nella directory di lavoro
COPY . /app

# Esponi la porta di Streamlit
EXPOSE 8501

# Comando per avviare l'applicazione
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]