import os.path
import http.server
import subprocess

import pytest
import requests
import time

here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        kwargs.pop("directory", None)
        super().__init__(*args, directory=here, **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isfile(path):
            return super().send_head()
        self.send_response(404)
        self.end_headers()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


@pytest.fixture(scope="session")
def server():
    P = subprocess.Popen(["python", __file__])
    timeout = 5
    url = "http://localhost:8000/azarr/tests/test.zarr"
    while True:
        try:
            assert requests.get(url + "/.zgroup").ok
            yield url
            break
        except Exception:
            time.sleep(0.1)
            timeout -= 0.1
            assert timeout > 0
    P.terminate()


if __name__ == "__main__":
    http.server.test(CORSRequestHandler)
