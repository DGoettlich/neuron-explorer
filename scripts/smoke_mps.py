"""One real-model request; stop the test server afterward."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import psutil


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('http://127.0.0.1:8792' + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def main():
    swap_start = psutil.swap_memory().used
    log_path = Path(tempfile.gettempdir()) / 'neuron-mps-smoke.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'neuron_compat.serve'],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited; see {log_path}')
                if psutil.virtual_memory().available < 512 * 1024**2 or psutil.swap_memory().used - swap_start > 128 * 1024**2 or shutil.disk_usage("/").free < 128 * 1024**2:
                    raise RuntimeError(f'Stopping smoke server: available RAM {psutil.virtual_memory().available / 1024**3:.2f} GiB; swap growth {(psutil.swap_memory().used - swap_start) / 1024**2:.0f} MiB; free disk {shutil.disk_usage("/").free / 1024**2:.0f} MiB')
                try:
                    request('/health')
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError(f'Server startup timed out; see {log_path}')
            result = json.loads(request('/v1/chat/completions', {
                'model': 'Qwen/Qwen3-0.6B',
                'messages': [{'role': 'user', 'content': 'Reply with exactly the word hello.'}],
                'temperature': 0, 'max_tokens': 8,
                'chat_template_kwargs': {'enable_thinking': False},
            }))
            answer = result['choices'][0]['message']['content']
            if answer.strip().rstrip('.!').lower() != 'hello':
                raise RuntimeError(f'Model returned an unexpected answer: {answer!r}')
            print(json.dumps({'model': 'Qwen/Qwen3-0.6B', 'answer': answer,
                              'usage': result.get('usage')}, indent=2))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    print(f'Server stopped. Log: {log_path}')


if __name__ == '__main__':
    main()
