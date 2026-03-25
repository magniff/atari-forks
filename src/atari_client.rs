use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

const VSOCK_GUEST_PORT: u16 = 5005;
const AGENT_READY_SENTINEL: &str = "AGENT_READY";

/// Host-side client for the Atari guest agent.
///
/// Connects via Firecracker's vsock AF_UNIX proxy. Watches serial
/// (Firecracker stdout) for AGENT_READY before connecting.
pub struct AtariClient {
    stream: UnixStream,
}

impl AtariClient {
    /// Wait for AGENT_READY on serial, then connect via vsock.
    ///
    /// `serial_stdout` is an iterator over lines from Firecracker's stdout.
    /// For restored VMs, pass `skip_ready = true` to skip the serial wait.
    pub fn connect(
        vsock_uds_path: &Path,
        serial_stdout: &mut Option<impl BufRead>,
        skip_ready: bool,
        timeout: Duration,
    ) -> Result<Self> {
        if !skip_ready {
            // Watch serial for AGENT_READY
            if let Some(reader) = serial_stdout.as_mut() {
                eprintln!("[debug] Waiting for AGENT_READY on serial...");
                let deadline = std::time::Instant::now() + timeout;
                let mut line = String::new();
                loop {
                    if std::time::Instant::now() > deadline {
                        bail!("Guest agent did not send AGENT_READY within {:?}", timeout);
                    }
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) => bail!("EOF on serial before AGENT_READY"),
                        Ok(_) => {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                eprintln!("[debug] serial: {}", &trimmed[..trimmed.len().min(100)]);
                            }
                            if line.contains(AGENT_READY_SENTINEL) {
                                eprintln!("[debug] Got AGENT_READY!");
                                break;
                            }
                        }
                        Err(e) => bail!("Error reading serial: {e}"),
                    }
                }
            } else {
                eprintln!("[debug] No serial handle, skipping AGENT_READY wait");
            }
        }

        eprintln!("[debug] Connecting to vsock at {:?}...", vsock_uds_path);
        let stream = Self::do_connect_vsock(vsock_uds_path, timeout)?;
        eprintln!("[debug] Vsock connected!");
        Ok(Self { stream })
    }

    pub fn connect_vsock(uds_path: &Path, timeout: Duration) -> Result<Self> {
        let stream = Self::do_connect_vsock(uds_path, timeout)?;
        Ok(Self { stream })
    }

    fn do_connect_vsock(uds_path: &Path, timeout: Duration) -> Result<UnixStream> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            match UnixStream::connect(uds_path) {
                Ok(mut sock) => {
                    sock.set_read_timeout(Some(Duration::from_secs(30)))?;
                    sock.set_write_timeout(Some(Duration::from_secs(5)))?;

                    // Firecracker vsock handshake: CONNECT <port>\n -> OK <port>\n
                    let cmd = format!("CONNECT {VSOCK_GUEST_PORT}\n");
                    sock.write_all(cmd.as_bytes())?;

                    let mut response = String::new();
                    let mut reader = BufReader::new(&sock);
                    reader.read_line(&mut response)?;

                    if response.starts_with("OK") {
                        // Reconstruct the stream from the reader
                        drop(reader);
                        return Ok(sock);
                    }
                }
                Err(_) => {}
            }

            if std::time::Instant::now() > deadline {
                bail!("Could not connect to guest via vsock at {:?}", uds_path);
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    }

    fn send_command(&mut self, cmd: &Value) -> Result<Value> {
        let msg = format!("{}\n", cmd);
        self.stream.write_all(msg.as_bytes())?;

        let mut reader = BufReader::new(&self.stream);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .context("read response from guest")?;

        let response: Value = serde_json::from_str(line.trim()).context("parse guest response")?;

        if response.get("status").and_then(|s| s.as_str()) == Some("error") {
            let msg = response
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            bail!("Guest error: {msg}");
        }

        Ok(response)
    }

    pub fn reset(&mut self) -> Result<Vec<u8>> {
        let resp = self.send_command(&serde_json::json!({"cmd": "reset"}))?;
        let obs_b64 = resp["obs"].as_str().context("missing obs field")?;
        let obs = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, obs_b64)?;
        Ok(obs)
    }

    pub fn step(&mut self, action: i64) -> Result<StepResult> {
        let resp = self.send_command(&serde_json::json!({"cmd": "step", "action": action}))?;
        let obs_b64 = resp["obs"].as_str().context("missing obs")?;
        let obs = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, obs_b64)?;
        Ok(StepResult {
            obs,
            reward: resp["reward"].as_f64().unwrap_or(0.0),
            done: resp["done"].as_bool().unwrap_or(false),
        })
    }

    pub fn get_legal_actions(&mut self) -> Result<Vec<i64>> {
        let resp = self.send_command(&serde_json::json!({"cmd": "get_actions"}))?;
        let actions = resp["actions"]
            .as_array()
            .context("missing actions")?
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        Ok(actions)
    }
}

pub struct StepResult {
    pub obs: Vec<u8>,
    pub reward: f64,
    pub done: bool,
}
