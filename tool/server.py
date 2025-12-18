#!/usr/bin/env python3
"""Simple socket server that sends scenario commands to clients and records raw received lines."""

import argparse
import datetime
import select
import sys
import logging
import os
import socket
import threading
from typing import Optional, Tuple


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000
RECEIVED_BASE_DIR = "received_trajectory"
SCENARIOS = ["scenario_1", "scenario_2", "scenario_3"]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# The JSONL playback functionality was removed per user request.


class SimpleRecorder:
    """Record raw lines received from the client into timestamped session directory."""

    def __init__(self, base_dir: str = RECEIVED_BASE_DIR) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()

    def record(self, raw_text: str) -> None:
        with self._lock:
            self._counter += 1
            filename = os.path.join(self.session_dir, f"message_{self._counter:05d}.txt")
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(raw_text.strip() + "\n")

    def finalize(self) -> None:
        logging.info("Saved %d received messages to %s", self._counter, self.session_dir)


def trajectory_receiver(
    conn: socket.socket,
    recorder: SimpleRecorder,
    stop_event: threading.Event,
) -> None:
    """Receive raw lines from the client and store them without JSON parsing."""
    buffer = b""
    while not stop_event.is_set():
        try:
            chunk = conn.recv(4096)
        except (ConnectionResetError, OSError) as exc:
            logging.warning("Receive loop terminating due to connection error: %s", exc)
            break
        if not chunk:
            logging.info("Client closed the connection")
            break
        # Show raw chunk bytes as decoded text (so it appears without b'' or quotes)
        if logging.getLogger().isEnabledFor(logging.INFO):
            try:
                decoded = chunk.decode("utf-8", errors="replace").strip()
                logging.info("%s", decoded)
            except Exception:
                try:
                    logging.info("%r", chunk)
                except Exception:
                    pass
        buffer += chunk
    stop_event.set()
    logging.info("Receiver thread exiting")


def handle_client(conn: socket.socket, addr: Tuple[str, int]) -> bool:
    """Handle a single client connection interactively.

    Operator can type 1/2/3 to send scenario_1/2/3 to the connected client.
    Returns True if server main loop should stop (operator typed 'exit'), False otherwise.
    """
    logging.info("Client connected from %s:%d", addr[0], addr[1])
    stop_event = threading.Event()
    recorder = SimpleRecorder()
    receiver = threading.Thread(target=trajectory_receiver, args=(conn, recorder, stop_event), daemon=True)
    receiver.start()

    server_should_stop = False
    prompt = "Enter 1/2/3 to send scenario, 'q' to close connection, 'exit' to stop server> "
    try:
        while not stop_event.is_set() and receiver.is_alive():
            # poll stdin so we can react to stop_event without blocking indefinitely
            rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
            if not rlist:
                continue
            line = sys.stdin.readline()
            if not line:
                # EOF
                break
            cmd = line.strip()
            if cmd == "":
                continue
            if cmd.lower() == "exit":
                server_should_stop = True
                logging.info("Operator requested server shutdown")
                break
            if cmd.lower() == "q":
                logging.info("Operator requested to close client connection")
                break
            if cmd in ("1", "2", "3"):
                scenario = SCENARIOS[int(cmd) - 1]
                try:
                    conn.sendall(scenario.encode("utf-8") + b"\n")
                    logging.info("Sent scenario '%s' to client %s:%d", scenario, addr[0], addr[1])
                except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                    logging.error("Failed to send scenario to client %s:%d: %s", addr[0], addr[1], exc)
                    stop_event.set()
                    break
            else:
                logging.warning("Unrecognized command: %s", cmd)
    except KeyboardInterrupt:
        logging.info("Operator interrupted; closing connection")
    finally:
        stop_event.set()
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()

    receiver.join(timeout=1.0)
    recorder.finalize()
    logging.info("Client handler finished for %s:%d", addr[0], addr[1])
    return server_should_stop


def accept_loop(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen()
        logging.info("Listening on %s:%d", host, port)
        while True:
            try:
                conn, addr = server_sock.accept()
            except KeyboardInterrupt:
                logging.info("Server interrupted, shutting down")
                break
            try:
                should_stop = handle_client(conn, addr)
            except Exception as exc:
                logging.exception("Error handling client %s:%d: %s", addr[0], addr[1], exc)
                should_stop = False
            if should_stop:
                logging.info("Shutting down server as requested by operator")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Socket server that sends scenario commands and records received lines.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Interface to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: %(default)s)")
    # No playback / directory options: JSONL playback removed
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    # Interactive single-client mode only: operator types 1/2/3 to send scenario lines to the connected client.
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    os.makedirs(RECEIVED_BASE_DIR, exist_ok=True)
    accept_loop(args.host, args.port)


if __name__ == "__main__":
    main()
