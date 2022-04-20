FROM gcc:10
ARG SRC_DIR=/usr/src/myapp
RUN mkdir -p ${SRC_DIR}
COPY hello.cpp ${SRC_DIR}
WORKDIR ${SRC_DIR}
RUN g++ -o hello hello.cpp
CMD ["./hello"]
