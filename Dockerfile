FROM python:3.11-slim

# Docling needs these for PDF rendering and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch — saves roughly 2GB over the default CUDA build
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# The editable install (`pip install -e .` below) runs before source code is
# copied into the image, so its package-discovery snapshot is always empty —
# setuptools finds nothing to map at that point. `streamlit run ui/app.py`
# executes the script directly and does not add /app to sys.path itself (unlike
# `python -c`/pytest, which implicitly do), so without this, top-level imports
# like `from core.config import Config` fail at runtime despite working under
# every other invocation style. PYTHONPATH sidesteps the editable-install
# mapping entirely and also means new top-level packages become importable
# immediately via the bind mount, with no rebuild required.
ENV PYTHONPATH=/app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
