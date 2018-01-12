import errno
import json
import logging
import os
import sys

import click

from .dist import RelayClient
from .es import run_master, run_worker, SharedNoiseTable


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)

def import_algo(name):
    if name == 'es':
        from . import es as algo
    elif name == 'ns-es' or name == "nsr-es":
        from . import nses as algo
    elif name == 'ga':
        from . import ga as algo
    elif name == 'rs':
        from . import rs as algo
    else:
        raise NotImplementedError()
    return algo

@cli.command()
@click.option('--algo')
@click.option('--exp_str')
@click.option('--exp_file')
@click.option('--master_socket_path', required=True)
@click.option('--log_dir')
def master(algo, exp_str, exp_file, master_socket_path, log_dir):
    # Start the master
    assert (exp_str is None) != (exp_file is None), 'Must provide exp_str xor exp_file to the master'
    if exp_str:
        exp = json.loads(exp_str)
    elif exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
    else:
        assert False
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)
    algo = import_algo(algo)
    algo.run_master({'unix_socket_path': master_socket_path}, log_dir, exp)


@cli.command()
@click.option('--algo')
@click.option('--master_host', required=True)
@click.option('--master_port', default=6379, type=int)
@click.option('--relay_socket_path', required=True)
@click.option('--num_workers', type=int, default=0)
def workers(algo, master_host, master_port, relay_socket_path, num_workers):
    # Start the relay
    master_redis_cfg = {'host': master_host, 'port': master_port}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    if os.fork() == 0:
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return
    # Start the workers
    algo = import_algo(algo)
    noise = algo.SharedNoiseTable()  # Workers share the same noise
    num_workers = num_workers if num_workers else os.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    for _ in range(num_workers):
        if os.fork() == 0:
            algo.run_worker(master_redis_cfg, relay_redis_cfg, noise=noise)
            return
    os.wait()


if __name__ == '__main__':
    cli()
