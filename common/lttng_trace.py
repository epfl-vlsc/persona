import lttng
import os
import getpass
import socket
import subprocess
import spur
import shlex
import multiprocessing.pool

class LTTngTracer:
    def __init__(self, trace_events, trace_output_dir, trace_name="bioflow_trace"):
        if type(trace_events) is str:
            trace_events = [trace_events]
        self.trace_events = trace_events
        user = getpass.getuser()
        trace_name = "{name}_{user}".format(name=trace_name, user=user)
        self.sess_name = trace_name
        domain = lttng.Domain()
        domain.type = lttng.DOMAIN_UST
        handle = lttng.Handle(domain=domain, session_name=trace_name)
        self.domain = domain
        self.handle = handle
        new_handle=trace_output_dir
        count = 0
        while os.path.exists(new_handle):
            new_handle = "{trace_dir}_{cnt}".format(trace_dir=trace_output_dir, cnt=count)
            count += 1
        trace_output_dir = new_handle
        os.makedirs(name=trace_output_dir, exist_ok=False)
        self.outdir = trace_output_dir

        curr_sessions = lttng.list_sessions()
        if trace_name in curr_sessions:
            lttng.destroy(trace_name) # note: this doesn't delete any files

    def _start_session(self):
        def make_event(nm):
            ev = lttng.Event()
            ev.name = nm
            ev.type = lttng.EVENT_TRACEPOINT
            ev.loglevel = lttng.EVENT_LOGLEVEL_ALL
            return ev

        events = (make_event(nm=name) for name in self.trace_events)
        r = lttng.create(name=self.sess_name, path=self.outdir)
        if r < 0:
            raise Exception("lttng.create({nm}) return code {code}".format(nm=self.sess_name, code=r))
        for ev in events:
            r = lttng.enable_event(handle=self.handle, event=ev, channel_name=None)
            if r < 0:
                raise Exception("lttng.enable_event({nm}) return code {code}".format(
                    nm=ev.name, code=r
                ))

    def __enter__(self):
        self._start_session()
        r = lttng.start(session_name=self.sess_name)
        if r < 0:
            raise Exception("lttng.start({nm}) return code {code}".format(nm=self.sess_name, code=r))

    def __exit__(self, exc_type, exc_val, exc_tb):
        r = lttng.stop(session_name=self.sess_name)
        if r < 0:
            raise Exception("lttng.stop({nm}) return code {code}".format(nm=self.sess_name, code=r))

        r = lttng.destroy(name=self.sess_name)
        if r < 0:
            raise Exception("lttng.destroy({nm}) return code {code}".format(nm=self.sess_name, code=r))

# Note that this assumes the user can log into the remote machine without a password
# please either make sure you can log into the remote machine without a password
class LTTngRemoteTracer:

    @staticmethod
    def verify_remotes(remotes):
        user = getpass.getuser()
        def make_socket(remote):
            shell = spur.SshShell(hostname=remote, username=user, missing_host_key=spur.ssh.MissingHostKey.accept)
            try:
                shell.run(shlex.split("""which lttng"""), allow_error=False)
            except spur.results.RunProcessError as rpe:
                print("lttng not found on machine {m}".format(m=remote))
                raise rpe

            #try:
            #    shell.run(shlex.split("""id | grep -q tracing"""), allow_error=False)
            #except spur.results.RunProcessError as rpe:
            #    print("Couldn't check for tracing group membership for '{usr}'. Got {exp}".format(usr=user, exp=rpe))
            #    raise rpe

            return shell

        remotes = set(remotes)
        sockets = [make_socket(remote=r) for r in remotes]
        return sockets

    def __init__(self, remote_addrs, trace_events, trace_output_dir, trace_name="bioflow_trace", server_timeout=5.0):
        if isinstance(remote_addrs, str):
            remote_addrs = (remote_addrs,)
        if isinstance(trace_events, str):
            trace_events = (trace_events,)
        self.trace_events = trace_events
        self.server_timeout = server_timeout
        self.remotes = self.verify_remotes(remotes=remote_addrs)
        self.remote_addrs = remote_addrs
        user = getpass.getuser()
        self.trace_name = "{trace}_{username}".format(trace=trace_name, username=user)
        self.hostname = "{}.iccluster.epfl.ch".format(socket.gethostname())
        new_out_dir = trace_output_dir
        count = 0
        while os.path.exists(new_out_dir):
            new_out_dir = "{td}_{cnt}".format(td=trace_output_dir, cnt=count)
            count += 1
        trace_output_dir = new_out_dir
        os.makedirs(name=trace_output_dir, exist_ok=False)
        self.trace_output_dir = trace_output_dir

    def start_server(self):
        # return the subprocess that runs the server
        # need to make the -o param to be pointed directly into the lttng dir
        if subprocess.call(shlex.split("pgrep lttng-relayd")) == 0:
            subprocess.check_call(shlex.split("sudo killall lttng-relayd"))
        server_proc = subprocess.Popen("lttng-relayd -o {trace_dir}".format(trace_dir=self.trace_output_dir), shell=True)
        if server_proc.poll():
            raise RuntimeError("Unable to start lttng-relayd in outdir '{outdir}'".format(outdir=self.trace_name))
        return server_proc

    # yes, all these try/except enters and exits are slow, but it helps see exactly which issue happened
    def start_remote(self, remote, address):
        result = remote.run(shlex.split("""pgrep lttng-sessiond"""), allow_error=True)
        if result.return_code != 0:
            print("{}: Starting lttng-sessiond".format(address))
            try:
                remote.run(shlex.split("""lttng-sessiond -d"""), allow_error=False)
            except spur.results.RunProcessError as rpe:
                print("{host}: Unable to start lttng-sessiond. Received '{err}'".format(err=rpe, host=address))
                raise rpe
        try:
            res = remote.run(shlex.split("""lttng list {trace_name}""".format(trace_name=self.trace_name)), allow_error=True)
            if res.return_code == 0:
                print("{host}: destroying old session '{sess}'".format(host=address, sess=self.trace_name))
                # allow error for true in case it is already stopped
                remote.run(shlex.split("""lttng stop {trace_name}""".format(trace_name=self.trace_name)), allow_error=True)
                remote.run(shlex.split("""lttng destroy {trace_name}""".format(trace_name=self.trace_name)), allow_error=False)
        except spur.results.RunProcessError as rpe:
            print("{host}: Couldn't destroy old session '{trace_name}' on remote. Got return {ret}".format(trace_name=self.trace_name, ret=rpe, host=address))
            raise rpe

        try:
            print("{host}: creating session '{sess}'".format(host=address, sess=self.trace_name))
            remote.run(shlex.split("""lttng create -U net://{svr} {trace_name}""".format(
                trace_name=self.trace_name, svr=self.hostname
            )), allow_error=False)
        except spur.results.RunProcessError as rpe:
            print("{host}: Couldn't create trace session '{name}' or remote '{remote}'".format(
                name=self.trace_name, remote=remote, host=address
            ))
            raise rpe

        for trace_event in self.trace_events:
            try:
                remote.run(shlex.split("""lttng enable-event -u {event_name}""".format(
                    event_name=trace_event )), allow_error=False)
            except spur.results.RunProcessError as rpe:
                print("{host}: Couldn't create trace event '{trace_event}' or remote '{remote}'".format(
                    trace_event=trace_event, remote=remote, host=address
                ))
                raise rpe
        try:
            print("{host}: starting session '{sess}'".format(host=address, sess=self.trace_name))
            remote.run(shlex.split("""lttng start {trace_name}""".format(trace_name=self.trace_name)), allow_error=False)
        except spur.results.RunProcessError as rpe:
            print("{host}: Couldn't start trace session '{name}' or remote '{remote}'".format(
                name=self.trace_name, remote=remote, host=address
            ))
            raise rpe

    def stop_remote(self, remote, address):
        #print("{host}: stopping session '{sess}'".format(host=address, sess=self.trace_name))
        ret_code = remote.run(shlex.split("""lttng stop {trace_name}""".format(trace_name=self.trace_name)), allow_error=True)
        # note that destroy doesn't delete the data
        if ret_code.return_code != 0:
            print("{host}: couldn't call `stop` on trace '{trace_name}'. Got return code {ret}".format(trace_name=self.trace_name, ret=ret_code, host=address))

        #print("{host}: destroying session '{sess}'".format(host=address, sess=self.trace_name))
        ret_code = remote.run(shlex.split("""lttng destroy {trace_name}""".format(trace_name=self.trace_name)), allow_error=True)
        if ret_code.return_code != 0:
            print("{host}: Couldn't call `destroy` on trace '{trace_name}'. Got return code {ret}".format(trace_name=self.trace_name, ret=ret_code, host=address))

    def __enter__(self):
        if len(self.trace_events) > 0:
            self.server = self.start_server()
            with multiprocessing.pool.ThreadPool(processes=len(self.remotes)) as pool:
                pool.starmap(self.start_remote, zip(self.remotes, self.remote_addrs))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.trace_events) > 0:
            with multiprocessing.pool.ThreadPool(processes=len(self.remotes)) as pool:
                pool.starmap(self.stop_remote, zip(self.remotes, self.remote_addrs))
            self.server.terminate()
            if self.server.wait(timeout=self.server_timeout) is None:
                self.server.kill()
