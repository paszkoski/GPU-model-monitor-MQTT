#!/usr/bin/env python3
"""
Helper script to enrich process data with actual OS start times, memory percentages
and — when the process belongs to a container — its container id and name.
Reads JSON from stdin and outputs enriched JSON to stdout.
Total GPU memory can be passed via GPU_MEMORY_TOTAL environment variable.

Container attribution needs two things, and degrades gracefully without either:
  * `pid: host` on this container, so /proc/<pid>/cgroup resolves the host pid;
  * the Docker socket mounted read-only, to turn the container id into its name.
With neither, every process simply reports container_id/container_name as null,
exactly as before this was added.
"""
import http.client
import json
import re
import socket
import sys
import os
import psutil
from datetime import datetime

# cgroup v2: "0::/../docker-<64 hex>.scope"; cgroup v1: ".../docker/<64 hex>";
# containerd/CRI: ".../cri-containerd-<64 hex>.scope".
# The leading (?:^|[/-]) matters: unanchored, any path segment merely ENDING in
# "docker" would match and yield a confident but wrong container id, which is worse
# than admitting we don't know.
_CONTAINER_ID_RE = re.compile(
    r'(?:^|/)(?:[a-z0-9]+-)?(?:docker|containerd)[-/]([0-9a-f]{64})'
)

DOCKER_SOCKET = os.getenv('DOCKER_SOCKET', '/var/run/docker.sock')

# id -> name, memoised for this invocation. The script is re-run once per sample,
# so this only spares repeat lookups within a single sweep; that is enough to keep
# the Docker API call count at one per distinct container rather than one per process.
_container_name_cache = {}

# Set once the socket itself proves unreachable, so a HUNG (not merely absent) Docker
# costs one timeout per invocation rather than one per distinct container. Without
# this, a host with several GPU-sharing containers could spend N x DOCKER_TIMEOUT
# serially and overrun the sampling interval — starving the data exactly when the
# host is already unhealthy. A 404 does NOT set this: that is a healthy socket
# answering about an unknown id.
_docker_unreachable = False

def _float_env(name, default):
    """Read a float from the environment, falling back on anything unusable.

    Deliberately total: this runs at import, and the caller keeps whatever we print
    on stdout. A raise here produces an EMPTY stdout — not even "[]" — which blanks
    the monitor's whole process table. An unset, blank (`- DOCKER_TIMEOUT` as a bare
    compose pass-through) or non-numeric value must therefore degrade to the default,
    never crash.
    """
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return float(default)


DOCKER_TIMEOUT = _float_env('DOCKER_TIMEOUT', 2.0)


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection over a unix socket — avoids a docker SDK dependency."""

    def __init__(self, socket_path, timeout=DOCKER_TIMEOUT):
        super().__init__('localhost', timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def get_container_id(pid):
    """Full container id for a pid, or None if it is not in a container.

    Requires `pid: host`; without it /proc/<pid> is this container's own namespace
    and the lookup simply misses.
    """
    try:
        with open('/proc/%s/cgroup' % pid, 'r') as fh:
            cgroup = fh.read()
    except (OSError, ValueError):
        return None
    match = _CONTAINER_ID_RE.search(cgroup)
    return match.group(1) if match else None


def get_container_name(container_id):
    """Resolve a container id to its name via the Docker socket, or None.

    Never raises: if the socket is absent, unreadable or slow, attribution is
    simply unavailable and the caller falls back to the id.
    """
    global _docker_unreachable

    if container_id in _container_name_cache:
        return _container_name_cache[container_id]
    if _docker_unreachable:
        # Socket already proved unreachable this invocation; don't pay the timeout again.
        return None

    name = None
    conn = None
    try:
        conn = _UnixHTTPConnection(DOCKER_SOCKET)
        conn.request('GET', '/containers/%s/json' % container_id)
        response = conn.getresponse()
        if response.status == 200:
            # Docker returns the name with a leading slash ("/VoiceStudio").
            name = (json.load(response).get('Name') or '').lstrip('/') or None
        else:
            # A non-200 means the socket is healthy and answered — e.g. 404 for an id
            # that has since exited. Don't mark Docker unreachable for that.
            response.read()
    except Exception:
        name = None
        _docker_unreachable = True
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    _container_name_cache[container_id] = name
    return name


def get_process_start_time(pid):
    """Get the actual OS start time of a process"""
    try:
        proc = psutil.Process(pid)
        start_time_unix = proc.create_time()
        start_time_str = datetime.fromtimestamp(start_time_unix).strftime('%Y-%m-%d %H:%M:%S')
        return start_time_str, start_time_unix
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return None, None

def format_lifetime(seconds):
    """Format lifetime in human-readable format"""
    if seconds <= 0:
        return "0s"
    
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def enrich_processes(processes_json, gpu_memory_total):
    """Enrich process data with actual OS start times and memory percentages.

    Always returns a valid JSON string. When the GPU has no processes the
    upstream query yields blank/non-JSON input; return an empty array ("[]")
    rather than echoing the bad input back, so downstream consumers
    (gpu_current_stats.json / mqtt_publisher) never receive malformed JSON.
    """
    if not processes_json or not processes_json.strip():
        return "[]"
    try:
        processes = json.loads(processes_json)
        if not isinstance(processes, list):
            return "[]"
        
        current_time = datetime.now().timestamp()
        
        for process in processes:
            pid = process.get('pid')
            if pid:
                start_time_str, start_time_unix = get_process_start_time(pid)
                if start_time_str:
                    process['process_start_time'] = start_time_str
                    actual_lifetime = int(current_time - start_time_unix)
                    process['actual_lifetime_seconds'] = actual_lifetime
                    process['lifetime_formatted'] = format_lifetime(actual_lifetime)
                else:
                    # Keep existing lifetime if we can't get actual start time
                    process['process_start_time'] = None
                    process['actual_lifetime_seconds'] = process.get('lifetime_seconds', 0)
                    process['lifetime_formatted'] = format_lifetime(process.get('lifetime_seconds', 0))
            
            # Attribute the process to its container, so consumers can say
            # "VoiceStudio is holding 16% of VRAM" rather than naming a binary
            # path like /opt/conda/bin/python3 that no operator recognises.
            #
            # Contained deliberately: the enclosing try/except returns "[]" for the
            # WHOLE sample, so an unanticipated failure here would blank the process
            # list rather than merely lose a name. Attribution is the least important
            # thing this script produces and must never cost the rest of it.
            try:
                container_id = get_container_id(pid) if pid else None
                process['container_id'] = container_id[:12] if container_id else None
                process['container_name'] = (
                    get_container_name(container_id) if container_id else None
                )
            except Exception:
                process['container_id'] = None
                process['container_name'] = None

            # Calculate memory percentage if total GPU memory is provided
            if gpu_memory_total > 0:
                process_memory = process.get('memory', 0) or 0
                memory_percent = round((process_memory / gpu_memory_total) * 100, 2)
                process['memory_percent'] = memory_percent

        return json.dumps(processes)
    except Exception as e:
        print(f"Error enriching processes: {e}", file=sys.stderr)
        return "[]"

if __name__ == '__main__':
    # Read total GPU memory from environment variable
    gpu_memory_total = float(os.getenv('GPU_MEMORY_TOTAL', '0'))
    
    # Read JSON from stdin
    input_json = sys.stdin.read()
    
    # Enrich and output
    enriched_json = enrich_processes(input_json, gpu_memory_total)
    print(enriched_json)

