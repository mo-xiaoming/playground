# https://github.com/JetBrains/clion-remote/blob/master/Dockerfile.remote-cpp-env
#
# CLion remote docker environment (How to build docker container, run and stop it)
#
# Build and run:
#   docker build -t clion/remote-cpp-env:0.5 -f Dockerfile.remote-cpp-env .
#   docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env clion/remote-cpp-env:0.5
#   ssh-keygen -f "$HOME/.ssh/known_hosts" -R "[localhost]:2222"
#
# stop:
#   docker stop clion_remote_env
#
# ssh credentials (test user):
#   user@password

FROM ubuntu:20.04

COPY ./sources.list /etc/apt/sources.list

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt-get update \
  && export RUN TZ=Asia/Shanghai && \
      apt-get install -y --no-install-recommends \
      ssh \
      build-essential \
      gcc-10 \
      g++-10 \
      gdb \
      clang \
      clang-format \
      clang-tidy \
      python3 \
      python3-pip \
      vim-nox \
      tar \
      curl \
      zip \
      unzip \
      rsync \
      git \
  && apt-get clean \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10

COPY ./.pip /root/.pip

RUN pip3 install cmake ninja

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m mx && yes 123456 | passwd mx

#RUN git clone https://gitee.com/mirrors/vcpkg.git && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics

ENV VCPKG_HOME=/vcpkg

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
