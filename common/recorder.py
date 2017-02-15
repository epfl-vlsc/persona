import psutil
import threading
import time
import spur
import sys
import multiprocessing.pool
import signal
import json
import getpass
import shlex

class UsageRecorder:

    meta_attrs_to_dict = ("gids", "uids", "ionice")
    meta_attrs = ("username", "terminal", "exe", "cwd", "create_time", "nice",
                  "name", "environ", "ppid", "pid", "cpu_affinity", "cmdline") + meta_attrs_to_dict

    sample_attrs_to_dict = ("cpu_times", "io_counters", "memory_full_info", "memory_info", "num_ctx_switches")
    sample_attrs_iter_to_dict = ("threads", "memory_maps")
    sample_attrs = ("num_threads", "num_fds", "open_files", "connections",
                    "memory_percent", "cpu_percent") + sample_attrs_to_dict + sample_attrs_iter_to_dict

    def __init__(self, meta_dict, pid=None, interval=1.0):
        assert interval > 0.0
        self.event_list = []
        self.stop_event = threading.Event()
        self.meta_dict = meta_dict
        self.interval = interval
        self.thread = threading.Thread(name="UsageRecorderThread", target=self._recorder_thread)
        if pid:
            self.process = psutil.Process(pid=pid)
        else:
            self.process = psutil.Process()
        proc_meta = self.process.as_dict(attrs=self.meta_attrs)
        for attrib in self.meta_attrs_to_dict:
            proc_meta[attrib] = dict(proc_meta[attrib]._asdict())

        meta_dict["process_meta"] = proc_meta

    def __enter__(self):
        self.start_clock = time.perf_counter()
        self.meta_dict["start_time"] = time.time()
        self.thread.start()
        return self

    def _append_instance(self):
        clock_time = time.perf_counter() - self.start_clock
        proc_sample = self.process.as_dict(attrs=self.sample_attrs)
        for attrib in self.sample_attrs_to_dict:
            proc_sample[attrib] = dict(proc_sample[attrib]._asdict())
        for attrib in self.sample_attrs_iter_to_dict:
            proc_sample[attrib] = [dict(a._asdict()) for a in proc_sample[attrib]]
        proc_sample["runtime"] = clock_time
        self.event_list.append(proc_sample)

    def _recorder_thread(self):
        while not self.stop_event.is_set() and self.process.is_running():
            self._append_instance()
            time.sleep(self.interval)
        if "stop_time" not in self.meta_dict:
            self.meta_dict["stop_time"] = time.time()
        if "runtime" not in self.meta_dict:
            self.meta_dict["runtime"] = time.perf_counter() - self.start_clock

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.meta_dict["stop_time"] = time.time()
        self.meta_dict["runtime"] = time.perf_counter() - self.start_clock
        self.meta_dict["samples"] = self.event_list
        self.event_list = [] # in case this gets called again
        self.stop_event.set()
        self.thread.join()

class RemoteRecorder:

    # may need a `head -n+3` call or something, or maybe it's `tail`
    _pidstat_cmd = "pidstat -hrdu -p {pid_list} 1 | sed '1d;/^[#]/{{4,$d}};/^[#]/s/^[#][ ]*//;/^$/d;s/^[ ]*//;s/[ ]\+/,/g'"

    def __init__(self, remote_user=getpass.getuser()):
        self.machine_map = {}
        self.process_map = {}
        self.shell_map = {}
        self.remote_user = remote_user
        self.result_map = {}

    def record_pids(self, machines_and_pids):
        """
        :param machines_and_pids: a dict of k = "machine.name",  v = [list or tuple of pids to record]
        """
        for hostname, pids in machines_and_pids.items():
            if hostname in self.machine_map:
                print("Warning: received duplicate entry for machine '{m}'".format(m=hostname), file=sys.stderr)
                self.machine_map[hostname].extend(pids)
            else:
                self.machine_map[hostname] = pids
                self.result_map[hostname] = ""

        def make_shell(h):
            return h, spur.SshShell(hostname=h, username=self.remote_user, missing_host_key=spur.ssh.MissingHostKey.accept)
        def spawn_pidstat(h, pids):
            remote = self.shell_map[h]
            pidstat_command = self._pidstat_cmd.format(
                pid_list=",".join(str(pid) for pid in pids)
            )
            print("Running pidstat command: '{}'".format(pidstat_command))

            return h, remote.spawn(["bash", "-c", pidstat_command], allow_error=False)
        with multiprocessing.pool.ThreadPool() as pool:
            self.shell_map = dict(pool.map(make_shell, self.machine_map.keys()))
            self.process_map = dict(pool.starmap(spawn_pidstat, self.machine_map.items()))

    def __enter__(self):
        """
        Actually start recording
        :return:
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        pidstat processes will die when all the processes in the list finish
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        def wait_for(hostname, pidstat_process):
            return hostname, pidstat_process.wait_for_result().output.decode()
        with multiprocessing.pool.ThreadPool() as pool:
            self.result_map = dict(pool.starmap(wait_for, self.process_map.items()))

    def get_result_map(self):
        return self.result_map

class NetworkRecorder:

    _clear_file_command = "[ ! -e {outfile} ] || sudo rm {outfile}"
    _outfile = "/tmp/sar_data"
    _sar_command = "sar -n DEV -o {outfile} 1 1>/dev/null" # redirect to null so we don't buffer unused output
    _sadf_command = "sadf -Ud -- {outfile} -n DEV" # we need to specify -n DEV again
    _check_install_command = "which sar || sudo apt-get install sysstat"

    def _prep_machine(self, machine):
        shell = spur.SshShell(hostname=machine, username=self.remote_user, missing_host_key=spur.ssh.MissingHostKey.accept)
        shell.run(["bash", "-c", self._check_install_command])
        shell.run(shlex.split("killall sadc"), allow_error=True) # no sadc = error!
        kill_cmd = self._clear_file_command.format(outfile=self._outfile)
        shell.run(["bash", "-c", kill_cmd])
        return machine, shell

    def _launch_sar_process(self, machine):
        shell = machine[1]
        # need to nuke this beforehand because it appends
        spawn_command = self._sar_command.format(outfile=self._outfile)
        return machine[0], machine[1].spawn(["bash", "-c", spawn_command], store_pid=True, allow_error=True)

    def _end_sar_process(self, process):
        hostname, sar_proc = process
        shell = self.machines[hostname]
        sar_proc.send_signal(signal=signal.SIGTERM)
        sar_proc.wait_for_result()
        shell.run(shlex.split("killall sadc"), allow_error=True) # no sadc = error!
        result = shell.run(shlex.split(self._sadf_command.format(outfile=self._outfile)))
        str_output = result.output.decode()
        default_host = shell.run(["bash", "-c", "ip r | grep default | awk '{{print $NF}}'"])
        default_host_res = default_host.output.decode()
        return hostname, {"default_host": default_host_res.strip(), "data": str_output}

    def __init__(self, machines, remote_user=getpass.getuser()):
        machines = set(machines)
        self.remote_user = remote_user
        self.sar_procs = {}
        self.results = {}
        with multiprocessing.pool.ThreadPool() as pool:
            self.machines = dict(pool.map(self._prep_machine, machines))

    def __enter__(self):
        with multiprocessing.pool.ThreadPool() as pool:
            self.sar_procs = dict(pool.map(self._launch_sar_process, self.machines.items()))

    def __exit__(self, exc_type, exc_val, exc_tb):
        with multiprocessing.pool.ThreadPool() as pool:
            self.results = dict(pool.map(self._end_sar_process, self.sar_procs.items()))
