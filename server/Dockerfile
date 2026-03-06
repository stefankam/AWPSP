FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------
# 1. Install system packages
# --------------------------
RUN apt-get update && apt-get install -y \
    git curl sudo python3 python3-pip python3-dev \
    openvswitch-switch \
    net-tools iproute2 iputils-ping \
    default-jre default-jdk \
    build-essential \
    lsb-release software-properties-common \
    && rm -rf /var/lib/apt/lists/*


# --------------------------
# 2. Install Python packages
# --------------------------
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt 

# --------------------------
# 3. Set working directory
# --------------------------
WORKDIR /app

# --------------------------
# 4. Copy only necessary files
# --------------------------
COPY main_server.py .
COPY requirements.txt .
COPY traces/ traces/
COPY topology_server.py .
COPY shared_state.py .	
COPY availability.py .
COPY experiment_suite.py .

# --------------------------
# 5. Set environment
# --------------------------
ENV FLASK_PORT=8080

EXPOSE 8080

# --------------------------
# 6. Run the server
# --------------------------
CMD ["python3", "-u", "main_server.py"]
