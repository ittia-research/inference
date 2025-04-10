ARG NIGHTLY_DATE="20250410"
ARG BASE_IMAGE="us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_tpuvm_$NIGHTLY_DATE"

FROM ${BASE_IMAGE}

# # Set up a non-root user (optional but recommended for security)
# TO-DO: Need privileged permissions ro access TPU devuices
# ARG USERNAME=user
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID
# RUN groupadd --gid $USER_GID $USERNAME && \
#     useradd -m -u $USER_UID -g $USER_GID -s /bin/bash $USERNAME

# USER $USERNAME
# WORKDIR /home/$USERNAME/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host 0.0.0.0", "--port 8000"]
