# Docker Setup for Author Switch Detector

## Quick Start

### 1. Build the Docker image

```bash
# CPU version (most common)
make build

# Or manually:
docker build -t author-switch-detector:latest .
```

Using a Docker File :

1) Building Files
docker compose build

- Or : make build

2) Start the application
- docker compose up app