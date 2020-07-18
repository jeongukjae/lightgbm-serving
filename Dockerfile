FROM python:2.7-alpine AS builder

WORKDIR /workspace
RUN apk --no-cache add git g++ cmake ninja libgomp
COPY . .
RUN git submodule update --init --recursive && \
    rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -GNinja && \
    ninja
RUN cd build && ./run-test

FROM alpine:3.12.0
LABEL maintainer="jeongukjae@gmail.com"

RUN apk --no-cache add musl libgcc libstdc++ libgomp
WORKDIR /
COPY --from=builder /workspace/build/lightgbm-server .
COPY --from=builder /workspace/third_party/LightGBM/lib_lightgbm.so /lib/.
ENTRYPOINT ["/lightgbm-server", "--host=0.0.0.0"]
