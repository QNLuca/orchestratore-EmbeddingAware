# Usa un'immagine base di Python ottimizzata
FROM python:3.10-slim

# Evita la creazione di file .pyc e garantisce il flush dell'output sui log
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# Installa dipendenze di sistema necessarie per compilare pacchetti come neal/dimod/tabu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt-get/lists/*

# Copia e installa i requisiti Python separatamente per sfruttare la cache di Docker
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia l'intero codice sorgente del progetto nella cartella del container
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
