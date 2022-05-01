FROM elasticsearch:7.3.1 as prod
RUN yum install https://repo.ius.io/ius-release-el7.rpm https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm -y
RUN yum update -y && \
    yum install -y \
        python36u \
        python36u-libs \
        python36u-devel \
        python36u-pip \
    && yum clean all
COPY data/ /app/data/
COPY src/ /app/src/
COPY requirements.txt /app/
WORKDIR /app/
RUN python3.6 -m pip install -r /app/requirements.txt
