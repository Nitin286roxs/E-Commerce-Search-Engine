FROM elasticsearch:7.3.1 as prod
RUN yum install https://repo.ius.io/ius-release-el7.rpm https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm -y
RUN yum update -y && \
    yum install -y \
        python36u \
        python36u-libs \
        python36u-devel \
        python36u-pip \
        git && \
    yum clean all
WORKDIR /usr/Nitin286roxs/
COPY requirements.txt /usr/
RUN git clone https://github.com/Nitin286roxs/E-Commerce-Search-Engine.git && \
    python3.6 -m pip install -r /usr/requirements.txt
