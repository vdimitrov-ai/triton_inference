FROM nvcr.io/nvidia/tritonserver:24.03-py3

WORKDIR /opt

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "tritonserver" ] 