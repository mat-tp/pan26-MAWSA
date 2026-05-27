#!/bin/bash

podman build -t author-switch-detector .

podman run -it \
-v "$(pwd)/dataset:/app/dataset:Z" \
localhost/author-switch-detector:latest