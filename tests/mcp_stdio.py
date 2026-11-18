"""A minimal MCP stdio client that drives a real server process in tests."""

import json
import os
import queue
import subprocess
import sys
import threading
import time

from legal_workspace_mcp.config import INDEX_PATH_ENV_VAR


class StdioServer:
    """One `python -m legal_workspace_mcp.server` process on newline-delimited JSON-RPC."""

    def __init__(self, workspace, index_path):
        env = os.environ.copy()
        env[INDEX_PATH_ENV_VAR] = str(index_path)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "legal_workspace_mcp.server", str(workspace)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, bufsize=1,
        )
        self._messages: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        self._next_id = 0
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self._messages.put(json.loads(line))

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr.append(line)

    def _send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=5.0):
        """Send a request; return (response, seconds). Raise TimeoutError if none arrives."""
        self._next_id += 1
        req_id = self._next_id
        started = time.monotonic()
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        deadline = started + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no answer to {method} in {timeout}s; exit={self.proc.poll()}; "
                    f"stderr tail: {''.join(self.stderr[-5:])}"
                )
            try:
                msg = self._messages.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg.get("id") == req_id:
                return msg, time.monotonic() - started

    def initialize(self, timeout=5.0):
        resp, elapsed = self.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        }, timeout=timeout)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp, elapsed

    def call_tool(self, name, arguments=None, timeout=20.0):
        """Call a tool and return its JSON text payload parsed."""
        resp, _ = self.request("tools/call", {"name": name, "arguments": arguments or {}},
                               timeout=timeout)
        assert "result" in resp, resp
        return json.loads(resp["result"]["content"][0]["text"])

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
