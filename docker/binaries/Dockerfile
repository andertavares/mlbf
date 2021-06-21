FROM ubuntu:18.04 as builder

ENV DOCKER_BUILDKIT=1

RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common \
    libboost-program-options-dev \
    gcc \
    g++ \
    make \
    cmake \
    git \
    zlib1g-dev \
    wget \
    make \
    libgmp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/github

# MIS
RUN git clone https://github.com/meelgroup/mis && \
    cd mis && \
    git submodule update --init && \
    make static && \
    cd ..

#    MARI
RUN wget https://bitbucket.org/malb/m4ri/downloads/m4ri-20200125.tar.gz && \
    tar -xvf m4ri-20200125.tar.gz && \
    cd m4ri-20200125 && \
    ./configure && \
    make && make install && \
    cd ../..

# CMS
RUN wget https://github.com/msoos/cryptominisat/archive/5.8.0.tar.gz && \
    tar -xvf 5.8.0.tar.gz && \
    cd cryptominisat-5.8.0 && \
    mkdir build && cd build && \
    cmake -DSTATICCOMPILE=ON .. && \
    make -j6 && make install && \
    cd ../..

# ApproxMC
RUN wget https://github.com/meelgroup/approxmc/archive/4.0.1.tar.gz && \
    tar -xvf 4.0.1.tar.gz && \
    cd  approxmc-4.0.1 && \
    mkdir build && cd build && \
    cmake -DSTATICCOMPILE=ON .. && \
    make -j6 && make install && \
    cd ../..

# Unigen3
RUN git clone https://github.com/meelgroup/unigen.git && \
    cd unigen && \
    mkdir build && cd build && \
    cmake -DSTATICCOMPILE=ON .. && \
    make -j6 && make install



FROM alpine:latest

WORKDIR /opt/binaries

COPY --from=builder /opt/github/mis/mis.py .
COPY --from=builder /opt/github/mis/togmus .
COPY --from=builder /opt/github/mis/muser2-dir/src/tools/muser2/muser2 .
COPY --from=builder /usr/local/bin/unigen ./unigen3

CMD ["/bin/sh"]
