//! TLS trust behaviour of the remote lookup client.
//!
//! `reqwest`'s `rustls-tls` feature verifies against the bundled webpki roots
//! only, so a lookup server presenting a self-signed certificate is rejected
//! with `UnknownIssuer` and the platform trust store cannot help. These tests
//! drive a real TLS listener with a freshly generated self-signed certificate
//! and pin the three outcomes the client must produce: rejected by default,
//! accepted when the certificate is supplied as an extra root, and accepted
//! when verification is explicitly disabled.

use std::io::Write as _;
use std::net::TcpListener as StdTcpListener;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use ntlmrain::remote_lookup::{RemoteLookupClient, RemoteLookupConfig, RemoteLookupError};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::rustls::pki_types::PrivateKeyDer;

const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// A TLS server whose certificate is self-signed for `localhost`. It answers
/// every request with `204 No Content`, so a successful `cancel` proves the
/// handshake was accepted and any error is a TLS failure rather than protocol
/// noise.
struct SelfSignedServer {
    base_url: String,
    cert_pem: String,
}

fn start_server() -> SelfSignedServer {
    let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
    let cert_pem = issued.cert.pem();
    let cert_der = issued.cert.der().clone();
    let key_der = PrivateKeyDer::try_from(issued.signing_key.serialize_der()).unwrap();

    // Bind synchronously so the port is known (and already listening) before
    // the test makes its first request.
    let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let (ready, started) = mpsc::channel();

    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let config = ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(vec![cert_der], key_der)
                .unwrap();
            let acceptor = TlsAcceptor::from(Arc::new(config));
            let listener = TcpListener::from_std(listener).unwrap();
            ready.send(()).unwrap();
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    continue;
                };
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let Ok(mut stream) = acceptor.accept(stream).await else {
                        return;
                    };
                    let mut buffer = [0u8; 4096];
                    let _ = stream.read(&mut buffer).await;
                    let _ = stream
                        .write_all(b"HTTP/1.1 204 No Content\r\nconnection: close\r\n\r\n")
                        .await;
                    let _ = stream.shutdown().await;
                });
            }
        });
    });

    started.recv_timeout(Duration::from_secs(10)).unwrap();
    SelfSignedServer {
        base_url: format!("https://localhost:{port}"),
        cert_pem,
    }
}

fn config(server: &SelfSignedServer) -> RemoteLookupConfig {
    RemoteLookupConfig {
        base_url: server.base_url.clone(),
        request_timeout: Duration::from_secs(10),
        ..RemoteLookupConfig::default()
    }
}

#[test]
fn self_signed_certificate_is_rejected_by_default() {
    let server = start_server();
    let client = RemoteLookupClient::new(config(&server)).unwrap();
    let error = client.cancel(TOKEN).unwrap_err();
    let message = error.to_string();
    assert!(
        matches!(error, RemoteLookupError::Connect(_)) && message.contains("UnknownIssuer"),
        "expected an untrusted-issuer connection failure, got: {message}"
    );
}

#[test]
fn certificate_authority_file_makes_the_server_trusted() {
    let server = start_server();
    let ca = tempfile::NamedTempFile::new().unwrap();
    ca.as_file().write_all(server.cert_pem.as_bytes()).unwrap();
    let client = RemoteLookupClient::new(RemoteLookupConfig {
        ca_certificate: Some(ca.path().to_path_buf()),
        ..config(&server)
    })
    .unwrap();
    client.cancel(TOKEN).unwrap();
}

#[test]
fn insecure_skips_verification_entirely() {
    let server = start_server();
    let client = RemoteLookupClient::new(RemoteLookupConfig {
        insecure: true,
        ..config(&server)
    })
    .unwrap();
    client.cancel(TOKEN).unwrap();
}

#[test]
fn an_unreadable_or_malformed_certificate_authority_is_reported() {
    let server = start_server();
    let missing = RemoteLookupClient::new(RemoteLookupConfig {
        ca_certificate: Some("/nonexistent/ca.pem".into()),
        ..config(&server)
    });
    assert!(matches!(
        missing,
        Err(RemoteLookupError::InvalidCaCertificate(_))
    ));

    let junk = tempfile::NamedTempFile::new().unwrap();
    junk.as_file().write_all(b"not a certificate").unwrap();
    let malformed = RemoteLookupClient::new(RemoteLookupConfig {
        ca_certificate: Some(junk.path().to_path_buf()),
        ..config(&server)
    });
    assert!(matches!(
        malformed,
        Err(RemoteLookupError::InvalidCaCertificate(_))
    ));
}
