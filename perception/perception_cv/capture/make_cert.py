"""Self-signed cert so phones will allow camera access (browsers need HTTPS off localhost).

    python -m perception_cv.capture.make_cert            # uses this machine's LAN IP
    python -m perception_cv.capture.make_cert 192.168.1.20
"""
import datetime
import ipaddress
import sys
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .netutil import local_ip


def main(ip: Optional[str] = None, out_dir: str = "certs") -> None:
    ip = ip or local_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(ip)), x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256()))
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    (out / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"wrote {out}/cert.pem and {out}/key.pem for {ip}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
