import time
import subprocess
import spur
import git
import sys
import os
import getpass
import collections
import multiprocessing.pool
import functools
import threading

Machine = collections.namedtuple(typename="Machine", field_names=("host", "shell"))

class WrapperWriter:
    def __init__(self, prefix, fd, stop_event):
        self.prefix = prefix
        self.fd = fd
        self.buff = ""
        self.stop_event = stop_event

    def write(self, output):
        out = output
        if isinstance(out, bytes):
            out = out.decode()
        self.buff += out
        if self.buff.endswith("\n") and not self.stop_event.is_set():
            self.fd.write("{pref}: {out}".format(pref=self.prefix, out=self.buff))
            self.buff=""

class ClusterManager:

    _server_script = "snap_align_local.py"
    _kill_script = "pgrep -a python3 | grep {script} | awk '{{print $1}}' | xargs kill -{level}"
    _find_script = "pgrep -a python3 | grep {script} | awk '{{print $1}}'"
    _sudo_kill_script = "pgrep -a python3 | grep {script} | awk '{{print $1}}' | xargs sudo kill -{level}"
    _kill_term = "TERM"
    _kill_kill = "KILL"

    def __init__(self, hosts, remote_prep_path, param_string, queue_host, wait_time=30, local_tensorflow_path=None, remote_user=getpass.getuser()):
        self.remote_prep_path = remote_prep_path
        self.hosts = hosts
        self.wait_time = wait_time
        self.remote_user = remote_user
        self.wrapper_stop_event = threading.Event() # automatically set to stop
        self.param_string = param_string
        self.queue_host = queue_host

        if local_tensorflow_path is None:
            self.local_tensorflow_repo = None
        else:
            # this will throw an exception if it doesn't exist
            self.local_tensorflow_repo = git.Repo(path=local_tensorflow_path)

    def _make_conn(self, host):
        return Machine(host=host, shell=spur.SshShell(hostname=host, username=self.remote_user, missing_host_key=spur.ssh.MissingHostKey.accept))

    def _kill_remote_servers(self, remote_machine, kill_level):
        shell = remote_machine.shell
        shell.run(["bash", "-c", self._kill_script.format(script=self._server_script, level=kill_level)], allow_error=True)
        # TODO actually check whether it was killed correctly, using TERM

    def _prep_machine(self, remote_machine, tensorflow_path, tf_align_path, tf_align_sha):
        host = remote_machine.host; shell = remote_machine.shell
        print("Prepping machine {host}...".format(host=host))

        if self.local_tensorflow_repo is not None:
            tf_sha = self.local_tensorflow_repo.head.object.hexsha
            command = "cd {tf_path} && git fetch origin && git checkout {tf_sha}".format(tf_path=tensorflow_path, tf_sha=tf_sha)
            print("Running: {}".format(command))
            shell.run(["bash", "-c", command])
            command = "cd {tf_path} && ./compile.sh".format(tf_path=tensorflow_path)
            print("Running: {}".format(command))
            shell.run(["bash", "-c", command])

        command = "cd {align_path} && git fetch origin && git checkout {align_sha}".format(align_path=tf_align_path, align_sha=tf_align_sha)
        print("Running: {}".format(command))
        shell.run(["bash", "-c", command])
        print("Done prepping machine {host}".format(host=host))

    def _spawn_machine(self, remote_machine, tensorflow_path, server_file, ceph_path, params, queue_host):
        host = remote_machine.host; shell = remote_machine.shell
        print("Running command in '{host}' : '{svr_path} {params} --queue-host {queue_host}'".format(host=host,
                                                                    svr_path=server_file, params=params, queue_host=queue_host))
        server_proc = shell.spawn(["bash", "-c", "source {tf_path}/python_dev/bin/activate && cd {ceph_path} && python3 {server_path} {params} --queue-host {queue_host} ".format(
            tf_path=tensorflow_path, server_path=server_file, ceph_path=ceph_path, params=params, queue_host=queue_host
        )], store_pid=True, stdout=WrapperWriter(prefix="{host}-out".format(host=host), fd=sys.stdout, stop_event=self.wrapper_stop_event),
                                  stderr=WrapperWriter(prefix="{host}-err".format(host=host), fd=sys.stderr, stop_event=self.wrapper_stop_event))
        time.sleep(self.wait_time)
        pid_output = shell.run(["bash", "-c", self._find_script.format(script=self._server_script)], allow_error=False).output.decode()
        real_pids = [int(str.encode(i)) for i in pid_output.split("\n") if len(i) != 0]
        return host, real_pids

    def _clean_machine(self, remote_machine):
        # we can't use the nice send_signal interface because of the relative path hack in the spawn call
        self._kill_remote_servers(remote_machine=remote_machine, kill_level=self._kill_term)

    def __enter__(self):
        if self.local_tensorflow_repo is not None:
            self.local_tensorflow_repo.remote().push()
            self.old_head = self.local_tensorflow_repo.head.ref

        tf_align_path = os.path.join(self.remote_prep_path, "persona-shell/modules/snap_align/")
        ceph_path = os.path.join(self.remote_prep_path, "agdutils/ceph_config")
        server_file = os.path.join(tf_align_path, self._server_script)
        tensorflow_path = os.path.join(self.remote_prep_path, "tensorflow-fpga")
        local_tf_align_repo = git.Repo(path=__file__, search_parent_directories=True)
        local_tf_align_repo.remote().push()
        tf_align_sha = local_tf_align_repo.head.object.hexsha
        params = self.param_string
        queue_host = self.queue_host

        old_tf_align_head = local_tf_align_repo.head.ref

        def mapper(a):
            idx, machine = a
            return self._spawn_machine(remote_machine=machine, tensorflow_path=tensorflow_path,
                                       server_file=server_file, ceph_path=ceph_path, params=params, queue_host=queue_host)
        def prep_map(a):
            self._prep_machine(remote_machine=a, tensorflow_path=tensorflow_path, tf_align_path=tf_align_path, tf_align_sha=tf_align_sha)

        with multiprocessing.pool.ThreadPool(processes=len(self.hosts)) as pool:
            self.cluster_shells = pool.map(self._make_conn, self.hosts)
            unique_remotes = { s.host: s for s in self.cluster_shells }
            pool.map(functools.partial(self._kill_remote_servers, kill_level=self._kill_kill), self.cluster_shells) # needs to run on all of them
            pool.map(prep_map, unique_remotes.values()) # only needs to run on the unique subset
            pidmap_list = pool.map(mapper, enumerate(self.cluster_shells))
            self.real_pids = dict(pidmap_list)
        old_tf_align_head.checkout() # happens when we're scanning through this machine as a remote
        return self.real_pids

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "old_head"):
            self.old_head.checkout()
        self.wrapper_stop_event.set()
        with multiprocessing.pool.ThreadPool() as pool:
            pool.map(self._clean_machine, self.cluster_shells)
