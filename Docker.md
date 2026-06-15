# Docker Setup for Author Switch Detector

## Quick Start

### 1. Build the Docker Image

Build the unified container environment:

```bash
docker compose build
```

Alternatively, if a Makefile is configured:

```bash
make build
```

You can also build manually:

```bash
docker build -t author-switch-detector:latest .
```

---

### 2. Run Services Locally

#### Run the Main Application / Training Pipeline

```bash
docker compose up app
```

#### Launch the Jupyter Environment

```bash
docker compose up jupyter
```

#### Run the Test Suite

```bash
docker compose up test
```

---

## Common Workflow

### Build Containers

```bash
docker compose build
```

or

```bash
make build
```

### Start the Application

```bash
docker compose up app
```

---

## TIRA Submission Testing

To test the exact execution format used by TIRA (which executes `run.sh`), run the container directly without Docker Compose:

### Build the Image

```bash
docker build -t author-switch-detector:latest .
```

### Run the Container

```bash
docker run --rm \
    -v ./data:/app/data \
    -v ./output:/output \
    author-switch-detector:latest \
    --input /app/data/raw \
    --output /output
```

This closely mirrors the environment used during TIRA evaluation.

---

# GPU & Docker Setup (Windows)

If you are setting up a new Windows machine and want Docker containers to access your NVIDIA GPU, complete the following steps.

## 1. Install the Latest NVIDIA Drivers

Do not rely on the default graphics drivers installed by Windows.

1. Visit the NVIDIA Driver Downloads page.
2. Download the latest driver for your GPU.
3. Install the driver.
4. Restart your computer if required.

---

## 2. Enable WSL2

Docker Desktop uses the Windows Subsystem for Linux (WSL2).

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart your computer when prompted.

After rebooting, create your Linux username and password when the WSL terminal opens.

---

## 3. Install Docker Desktop

1. Download Docker Desktop for Windows.
2. Run the installer.
3. Ensure **"Use the WSL 2 based engine"** is enabled.
4. Open Docker Desktop.
5. Navigate to:

```
Settings → General
```

6. Verify that **Use the WSL 2 based engine** is checked.

---

## 4. Verify GPU Access from Docker

Run the following command:

```powershell
docker compose run --rm app python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

Expected output:

```text
GPU Available: True
```

If you are using TensorFlow instead of PyTorch:

```powershell
docker compose run --rm app python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Troubleshooting

If GPU detection returns `False`:

1. Confirm that the latest NVIDIA drivers are installed.
2. Restart Docker Desktop.
3. Restart WSL:

```powershell
wsl --shutdown
```

4. Start Docker Desktop again.
5. Re-run the GPU verification command.

If problems persist, verify that your GPU is visible from Windows:

```powershell
nvidia-smi
```

If `nvidia-smi` does not work, the issue is likely related to the NVIDIA driver installation rather than Docker.
