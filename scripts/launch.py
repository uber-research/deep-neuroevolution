import datetime
import json
import os

import click

AMI_MAP = {
    "us-west-1": "FILL IN YOUR AMI HERE",
}


def highlight(x):
    if not isinstance(x, str):
        x = json.dumps(x, sort_keys=True, indent=2)
    click.secho(x, fg='green')


def upload_archive(exp_name, archive_excludes, s3_bucket):
    import hashlib, os.path as osp, subprocess, tempfile, uuid, sys

    # Archive this package
    thisfile_dir = osp.dirname(osp.abspath(__file__))
    pkg_parent_dir = osp.abspath(osp.join(thisfile_dir, '..', '..'))
    pkg_subdir = osp.basename(osp.abspath(osp.join(thisfile_dir, '..')))
    assert osp.abspath(__file__) == osp.join(pkg_parent_dir, pkg_subdir, 'scripts', 'launch.py'), 'You moved me!'

    # Run tar
    tmpdir = tempfile.TemporaryDirectory()
    local_archive_path = osp.join(tmpdir.name, '{}.tar.gz'.format(uuid.uuid4()))
    tar_cmd = ["tar", "-zcvf", local_archive_path, "-C", pkg_parent_dir]
    for pattern in archive_excludes:
        tar_cmd += ["--exclude", pattern]
    tar_cmd += ["-h", pkg_subdir]
    highlight(" ".join(tar_cmd))

    if sys.platform == 'darwin':
        # Prevent Mac tar from adding ._* files
        env = os.environ.copy()
        env['COPYFILE_DISABLE'] = '1'
        subprocess.check_call(tar_cmd, env=env)
    else:
        subprocess.check_call(tar_cmd)

    # Construct remote path to place the archive on S3
    with open(local_archive_path, 'rb') as f:
        archive_hash = hashlib.sha224(f.read()).hexdigest()
    remote_archive_path = '{}/{}_{}.tar.gz'.format(s3_bucket, exp_name, archive_hash)

    # Upload
    upload_cmd = ["aws", "s3", "cp", local_archive_path, remote_archive_path]
    highlight(" ".join(upload_cmd))
    subprocess.check_call(upload_cmd)

    presign_cmd = ["aws", "s3", "presign", remote_archive_path, "--expires-in", str(60 * 60 * 24 * 30)]
    highlight(" ".join(presign_cmd))
    remote_url = subprocess.check_output(presign_cmd).decode("utf-8").strip()
    return remote_url


def make_disable_hyperthreading_script():
    return """
# disable hyperthreading
# https://forums.aws.amazon.com/message.jspa?messageID=189757
for cpunum in $(
    cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list |
    sed 's/-/,/g' | cut -s -d, -f2- | tr ',' '\n' | sort -un); do
        echo 0 > /sys/devices/system/cpu/cpu$cpunum/online
done
"""


def make_download_and_run_script(code_url, cmd):
    return """su -l ubuntu <<'EOF'
set -x
cd ~
wget --quiet "{code_url}" -O code.tar.gz
tar xvaf code.tar.gz
rm code.tar.gz
cd es-distributed
{cmd}
EOF
""".format(code_url=code_url, cmd=cmd)


def make_master_script(code_url, exp_str):
    cmd = """
cat > ~/experiment.json <<< '{exp_str}'
python -m es_distributed.main master \
    --master_socket_path /var/run/redis/redis.sock \
    --log_dir ~ \
    --exp_file ~/experiment.json
    """.format(exp_str=exp_str)
    return """#!/bin/bash
{
set -x

%s

# Disable redis snapshots
echo 'save ""' >> /etc/redis/redis.conf

# Make the unix domain socket available for the master client
# (TCP is still enabled for workers/relays)
echo "unixsocket /var/run/redis/redis.sock" >> /etc/redis/redis.conf
echo "unixsocketperm 777" >> /etc/redis/redis.conf
mkdir -p /var/run/redis
chown ubuntu:ubuntu /var/run/redis

systemctl restart redis

%s
} >> /home/ubuntu/user_data.log 2>&1
""" % (make_disable_hyperthreading_script(), make_download_and_run_script(code_url, cmd))


def make_worker_script(code_url, master_private_ip):
    cmd = ("MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
           "python -m es_distributed.main workers "
           "--master_host {} "
           "--relay_socket_path /var/run/redis/redis.sock").format(master_private_ip)
    return """#!/bin/bash
{
set -x

%s

# Disable redis snapshots
echo 'save ""' >> /etc/redis/redis.conf

# Make redis use a unix domain socket and disable TCP sockets
sed -ie "s/port 6379/port 0/" /etc/redis/redis.conf
echo "unixsocket /var/run/redis/redis.sock" >> /etc/redis/redis.conf
echo "unixsocketperm 777" >> /etc/redis/redis.conf
mkdir -p /var/run/redis
chown ubuntu:ubuntu /var/run/redis

systemctl restart redis

%s
} >> /home/ubuntu/user_data.log 2>&1
""" % (make_disable_hyperthreading_script(), make_download_and_run_script(code_url, cmd))


@click.command()
@click.argument('exp_files', nargs=-1, type=click.Path(), required=True)
@click.option('--key_name', default=lambda: os.environ["KEY_NAME"])
@click.option('--aws_access_key_id', default=os.environ.get("AWS_ACCESS_KEY", None))
@click.option('--aws_secret_access_key', default=os.environ.get("AWS_ACCESS_SECRET", None))
@click.option('--archive_excludes', default=(".git", "__pycache__", ".idea", "scratch"))
@click.option('--s3_bucket')
@click.option('--spot_price')
@click.option('--region_name')
@click.option('--zone')
@click.option('--cluster_size', type=int, default=1)
@click.option('--spot_master', is_flag=True, help='Use a spot instance as the master')
@click.option('--master_instance_type')
@click.option('--worker_instance_type')
@click.option('--security_group')
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def main(exp_files,
         key_name,
         aws_access_key_id,
         aws_secret_access_key,
         archive_excludes,
         s3_bucket,
         spot_price,
         region_name,
         zone,
         cluster_size,
         spot_master,
         master_instance_type,
         worker_instance_type,
         security_group,
         yes
         ):

    highlight('Launching:')
    highlight(locals())

    import boto3
    ec2 = boto3.resource(
        "ec2",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    as_client = boto3.client(
        'autoscaling',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    for i_exp_file, exp_file in enumerate(exp_files):
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
        highlight('Experiment [{}/{}]:'.format(i_exp_file + 1, len(exp_files)))
        highlight(exp)
        if not yes:
            click.confirm('Continue?', abort=True)

        exp_prefix = exp['exp_prefix']
        exp_str = json.dumps(exp)

        exp_name = '{}_{}'.format(exp_prefix, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        code_url = upload_archive(exp_name, archive_excludes, s3_bucket)
        highlight("code_url: " + code_url)

        image_id = AMI_MAP[region_name]
        highlight('Using AMI: {}'.format(image_id))

        if spot_master:
            import base64
            requests = ec2.meta.client.request_spot_instances(
                SpotPrice=spot_price,
                InstanceCount=1,
                LaunchSpecification=dict(
                    ImageId=image_id,
                    KeyName=key_name,
                    InstanceType=master_instance_type,
                    EbsOptimized=True,
                    SecurityGroups=[security_group],
                    Placement=dict(
                        AvailabilityZone=zone,
                    ),
                    UserData=base64.b64encode(make_master_script(code_url, exp_str).encode()).decode()
                )
            )['SpotInstanceRequests']
            assert len(requests) == 1
            request_id = requests[0]['SpotInstanceRequestId']
            # Wait for fulfillment
            highlight('Waiting for spot request {} to be fulfilled'.format(request_id))
            ec2.meta.client.get_waiter('spot_instance_request_fulfilled').wait(SpotInstanceRequestIds=[request_id])
            req = ec2.meta.client.describe_spot_instance_requests(SpotInstanceRequestIds=[request_id])
            master_instance_id = req['SpotInstanceRequests'][0]['InstanceId']
            master_instance = ec2.Instance(master_instance_id)
        else:
            master_instance = ec2.create_instances(
                ImageId=image_id,
                KeyName=key_name,
                InstanceType=master_instance_type,
                EbsOptimized=True,
                SecurityGroups=[security_group],
                MinCount=1,
                MaxCount=1,
                Placement=dict(
                    AvailabilityZone=zone,
                ),
                UserData=make_master_script(code_url, exp_str)
            )[0]
        master_instance.create_tags(
            Tags=[
                dict(Key="Name", Value=exp_name + "-master"),
                dict(Key="es_dist_role", Value="master"),
                dict(Key="exp_prefix", Value=exp_prefix),
                dict(Key="exp_name", Value=exp_name),
            ]
        )
        highlight("Master created. IP: %s" % master_instance.public_ip_address)

        config_resp = as_client.create_launch_configuration(
            ImageId=image_id,
            KeyName=key_name,
            InstanceType=worker_instance_type,
            EbsOptimized=True,
            SecurityGroups=[security_group],
            LaunchConfigurationName=exp_name,
            UserData=make_worker_script(code_url, master_instance.private_ip_address),
            SpotPrice=spot_price,
        )
        assert config_resp["ResponseMetadata"]["HTTPStatusCode"] == 200

        asg_resp = as_client.create_auto_scaling_group(
            AutoScalingGroupName=exp_name,
            LaunchConfigurationName=exp_name,
            MinSize=cluster_size,
            MaxSize=cluster_size,
            DesiredCapacity=cluster_size,
            AvailabilityZones=[zone],
            Tags=[
                dict(Key="Name", Value=exp_name + "-worker"),
                dict(Key="es_dist_role", Value="worker"),
                dict(Key="exp_prefix", Value=exp_prefix),
                dict(Key="exp_name", Value=exp_name),
            ]
            # todo: also try placement group to see if there is increased networking performance
        )
        assert asg_resp["ResponseMetadata"]["HTTPStatusCode"] == 200
        highlight("Scaling group created")

        highlight("%s launched successfully." % exp_name)
        highlight("Manage at %s" % (
            "https://%s.console.aws.amazon.com/ec2/v2/home?region=%s#Instances:sort=tag:Name" % (
            region_name, region_name)
        ))


if __name__ == '__main__':
    main()
