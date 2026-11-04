"""Small release tests: launcher contracts and actual Uvicorn shutdown/restart.

These use an explicitly fake predictor, not a GPU or browser demo claim.
"""
import contextlib
import io
import socket
import threading
import time
import unittest
from unittest.mock import patch

import httpx
import uvicorn

from dashboard import create_app
from dashboard_service import API_VERSION, DashboardRuntime, load_config
import launch_dashboard
from test_dashboard import FakePredictor


class ReleaseTests(unittest.TestCase):
    def test_launcher_reuses_ready_service_without_new_process(self):
        config = load_config()
        status = {"api_version":API_VERSION,"model_state": "ready", "model": {"checkpoint_sha256": config["checkpoint_sha256"]}}
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            port = listener.getsockname()[1]
            with patch("sys.argv", ["launch_dashboard.py", "--port", str(port), "--no-browser"]), \
                 patch("launch_dashboard.preflight", return_value=config), \
                 patch("launch_dashboard.read_status", return_value=status), \
                 patch("launch_dashboard.subprocess.Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch_dashboard.main(), 0)
                popen.assert_not_called()

    def test_launcher_reports_port_conflict_and_invalid_initialization(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            with patch("sys.argv", ["launch_dashboard.py", "--port", str(listener.getsockname()[1]), "--no-browser"]), \
                 patch("launch_dashboard.preflight", return_value=load_config()), \
                 patch("launch_dashboard.read_status", return_value={"model_state": "loading", "model": None}), \
                 contextlib.redirect_stderr(io.StringIO()) as error, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch_dashboard.main(), 1)
                self.assertIn("occupied", error.getvalue())
        with patch("sys.argv", ["launch_dashboard.py", "--check"]), \
             patch("launch_dashboard.preflight", side_effect=RuntimeError("Missing dependency fixture")), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(launch_dashboard.main(), 1)
            self.assertIn("Missing dependency", error.getvalue())

    def test_real_uvicorn_clean_shutdown_and_restart(self):
        config = load_config()
        for _ in range(2):
            runtime = DashboardRuntime(config, FakePredictor)
            runtime.predictor, runtime.model_state = FakePredictor(), "ready"
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0)); listener.listen()
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(create_app(config, runtime), log_level="error", access_log=False))
                worker = threading.Thread(target=lambda: server.run(sockets=[listener]))
                worker.start()
                try:
                    deadline = time.monotonic() + 5
                    while not server.started and time.monotonic() < deadline: time.sleep(.01)
                    self.assertTrue(server.started)
                    state = httpx.get(f"http://127.0.0.1:{port}/api/status").raise_for_status().json()
                    self.assertEqual(state["model_state"], "ready")
                finally:
                    server.should_exit = True; worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertIsNone(runtime.session)


if __name__ == "__main__": unittest.main()
