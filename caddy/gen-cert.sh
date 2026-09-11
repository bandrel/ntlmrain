#!/bin/sh
# Generates the self-signed certificate the proxy serves by default.
#
# Runs to completion before Caddy starts and does nothing if a certificate is
# already present, so restarting the stack does not hand out a new identity
# every time and invalidate the trust decision you already made in the browser.
#
# Not used in Let's Encrypt mode: setting TLS_ARGS to an email address makes
# Caddy manage its own certificate and ignore these files.
set -eu

CRT=/certs/tls.crt
KEY=/certs/tls.key

if [ -s "$CRT" ] && [ -s "$KEY" ]; then
    echo "cert-init: certificate already present at $CRT, leaving it alone"
    openssl x509 -in "$CRT" -noout -subject -dates -ext subjectAltName 2>/dev/null || true
    exit 0
fi

command -v openssl >/dev/null 2>&1 || apk add --no-cache openssl >/dev/null

# Browsers ignore CN and match on subjectAltName only, so every name or
# address you intend to reach this host by has to be listed here.
SAN="DNS:localhost,IP:127.0.0.1"

for host in $(printf '%s' "${CERT_HOSTS:-}" | tr ',' ' '); do
    [ -n "$host" ] || continue
    case "$host" in
        # Bare IPv4/IPv6 literals must be IP SANs; a DNS SAN with an address
        # in it does not match when the browser connects to that address.
        *:*)        SAN="$SAN,IP:$host" ;;
        *[!0-9.]*)  SAN="$SAN,DNS:$host" ;;
        *)          SAN="$SAN,IP:$host" ;;
    esac
done

echo "cert-init: generating self-signed certificate for $SAN"
openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
    -keyout "$KEY" -out "$CRT" \
    -subj "/CN=ntlmrain" \
    -addext "subjectAltName=$SAN" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth"

chmod 644 "$CRT"
chmod 644 "$KEY"
echo "cert-init: done"
openssl x509 -in "$CRT" -noout -subject -dates -ext subjectAltName
