use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

const VSOCK_GUEST_PORT: u16 = 5005;

/// Host-side client for the Atari guest agent.
///
/// Connects via Firecracker's vsock AF_UNIX proxy. Watches serial
/// (Firecracker stdout) for AGENT_READY before connecting.
pub struct AtariClient {
    stream: UnixStream,
}

impl AtariClient {
    pub fn connect_vsock(uds_path: &Path, timeout: Duration) -> Result<Self> {
        let stream = Self::do_connect_vsock(uds_path, timeout)?;
        Ok(Self { stream })
    }

    fn do_connect_vsock(uds_path: &Path, timeout: Duration) -> Result<UnixStream> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            match UnixStream::connect(uds_path) {
                Ok(sock) => {
                    sock.set_read_timeout(Some(Duration::from_secs(30)))?;
                    sock.set_write_timeout(Some(Duration::from_secs(5)))?;

                    // Firecracker vsock handshake: CONNECT <port>\n -> OK <port>\n
                    let cmd = format!("CONNECT {VSOCK_GUEST_PORT}\n");
                    (&sock).write_all(cmd.as_bytes())?;

                    let mut response = String::new();
                    let mut reader = BufReader::new(&sock);
                    reader.read_line(&mut response)?;

                    if response.starts_with("OK") {
                        drop(reader);
                        return Ok(sock);
                    }
                    // Handshake failed — retry
                }
                Err(_) => {}
            }

            if std::time::Instant::now() > deadline {
                bail!("Could not connect to guest via vsock at {:?}", uds_path);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn send_command(&mut self, cmd: &Value) -> Result<Value> {
        let msg = format!("{}\n", cmd);
        self.stream.write_all(msg.as_bytes())?;

        let mut reader = BufReader::new(&self.stream);
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .context("read response from guest")?;

        if bytes_read == 0 {
            bail!("The vm has exited");
        }

        let response: Value = serde_json::from_str(line.trim())
            .with_context(|| format!("parse guest response: {:?}", line.trim()))?;

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
