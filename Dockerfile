FROM python:3.6-stretch

# Install dependencies: tmux, wget, and redis
RUN apt-get update && \
    apt-get install -y tmux wget cmake && \
    wget http://download.redis.io/redis-stable.tar.gz && \
    tar xvzf redis-stable.tar.gz && \
    cd redis-stable && \
    make && \
    make install

ADD . /root/deep-neuroevolution/

# Install python dependencies directly
# virutalenv doesn't make sense in a docker container 
RUN cd ~/deep-neuroevolution/ && \
    pip install -r requirements.txt

CMD bash -c "cd /root/deep-neuroevolution && ./scripts/local_run_redis.sh"


