"""Exercise real TLS connections, not just socket-family selection."""

import asyncio
from datetime import datetime, timedelta, timezone
import http.client
import ipaddress
import logging
import socket
import ssl
import threading
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI
import pytest

from roborock_local_server.backend import MqttTlsProxy, RuntimeCredentialsStore
from roborock_local_server.server import ManagedFastApiServer


@pytest.fixture
def tls_files(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address("::1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file = tmp_path / "cert.pem"
    key_file = tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ))
    return cert_file, key_file


def require_ipv6():
    try:
        with socket.create_server(("::1", 0), family=socket.AF_INET6):
            pass
    except OSError as exc:
        pytest.skip(f"IPv6 loopback is unavailable: {exc}")


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_https_tls_listener(host, tls_files):
    if ":" in host:
        require_ipv6()
    cert_file, key_file = tls_files
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    async def exercise():
        server = ManagedFastApiServer(
            app=app, bind_host=host, port=0, tls_enabled=True, cert_file=cert_file, key_file=key_file
        )
        await server.start()
        try:
            port = server._server.servers[0].sockets[0].getsockname()[1]

            def request():
                conn = http.client.HTTPSConnection(
                    host, port, timeout=3, context=ssl.create_default_context(cafile=str(cert_file))
                )
                try:
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    assert response.status == 200
                    assert response.read() == b'{"ok":true}'
                finally:
                    conn.close()

            await asyncio.to_thread(request)
        finally:
            await server.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize("listen_host,backend_host", [
    ("127.0.0.1", "127.0.0.1"),
    ("::1", "127.0.0.1"),
    ("::1", "::1"),
    ("::", "127.0.0.1"),
])
def test_mqtt_tls_relay(listen_host, backend_host, tls_files, tmp_path):
    if ":" in listen_host or ":" in backend_host:
        require_ipv6()
    cert_file, key_file = tls_files
    family = socket.AF_INET6 if ":" in backend_host else socket.AF_INET
    backend_listener = socket.create_server((backend_host, 0), family=family)
    backend_listener.settimeout(3)
    backend_port = backend_listener.getsockname()[1]
    connect_packet = bytes.fromhex("101300044d51545404c2003c000163000175000170")
    connack = b"\x20\x02\x00\x00"
    received = []

    def serve_backend():
        with backend_listener:
            conn, _ = backend_listener.accept()
            with conn:
                conn.settimeout(3)
                packet = bytearray()
                while len(packet) < len(connect_packet):
                    chunk = conn.recv(len(connect_packet) - len(packet))
                    if not chunk:
                        break
                    packet.extend(chunk)
                received.append(bytes(packet))
                conn.sendall(connack)

    backend_thread = threading.Thread(target=serve_backend, daemon=True)
    backend_thread.start()
    credentials = RuntimeCredentialsStore(tmp_path / "credentials.json")
    credentials.update_base(mqtt_usr="u", mqtt_passwd="p", mqtt_clientid="c")
    proxy = MqttTlsProxy(
        cert_file=cert_file, key_file=key_file, listen_host=listen_host, listen_port=0,
        backend_host=backend_host, backend_port=backend_port, localkey="test-local-key",
        logger=logging.getLogger("test.ipv6"), decoded_jsonl=tmp_path / "mqtt.jsonl",
        runtime_credentials=credentials,
    )
    proxy_thread = proxy.start()
    try:
        deadline = time.monotonic() + 3
        while not proxy._running and proxy_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert proxy._running, "MQTT listener did not start"
        port = proxy._server_socket.getsockname()[1]
        client_host = "::1" if listen_host == "::" else listen_host
        context = ssl.create_default_context(cafile=str(cert_file))
        with socket.create_connection((client_host, port), timeout=3) as raw:
            with context.wrap_socket(raw, server_hostname=client_host) as client:
                client.sendall(connect_packet)
                assert client.recv(4) == connack
        backend_thread.join(timeout=3)
        assert not backend_thread.is_alive()
        assert received == [connect_packet]
    finally:
        proxy.stop()
        proxy_thread.join(timeout=3)
        backend_listener.close()
    assert not proxy_thread.is_alive(), "Stopping the listener must release accept()"
