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

- Building :
  bash` docker build -t author-switch-detector .`

- Run the Application :
  bash```
  docker run --rm \
   -v "$(pwd)/dataset:/app/dataset" \
  -v "$(pwd)/outputs:/app/outputs" \
   author-switch-detector \
   python -m src.main --mode train

  ```

  ```
