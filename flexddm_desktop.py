from __future__ import annotations

import socket
import threading
import webbrowser
from http.server import ThreadingHTTPServer

from flexddm.app import FlexDDMHandler


def _close_splash() -> None:
    try:
        import pyi_splash

        pyi_splash.close()
    except Exception:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), FlexDDMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"

    try:
        import webview

        window = webview.create_window("FlexDDM", url, width=1320, height=900, min_size=(980, 680))
        _close_splash()
        webview.start()
        _ = window
    except Exception:
        _close_splash()
        webbrowser.open(url)
        try:
            while thread.is_alive():
                thread.join(timeout=1.0)
        except KeyboardInterrupt:
            pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
