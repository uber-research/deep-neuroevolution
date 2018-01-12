#!/bin/sh
tmux new -s redis -d
tmux send-keys -t redis 'redis-server redis_config/redis_master.conf' C-m
tmux split-window -t redis
tmux send-keys -t redis 'redis-server redis_config/redis_local_mirror.conf' C-m
tmux a -t redis
