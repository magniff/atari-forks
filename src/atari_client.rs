use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

const VSOCK_GUEST_PORT: u16 = 5005;

/// Host-side client for the Atari guest agent.
///
/// Connects via Firecracker's vsock AF_UNIX proxy. Uses a persistent
/// BufReader to avoid losing data when the guest sends responses
/// faster than we read them (the old code created a new BufReader per
/// call, silently dropping any buffered bytes).
pub struct AtariClient {
    reader: BufReader<tokio::io::ReadHalf<UnixStream>>,
    writer: tokio::io::WriteHalf<UnixStream>,
}

impl AtariClient {
    /// Connect to the guest agent via the vsock Unix socket.
    ///
    /// Performs the Firecracker vsock handshake (CONNECT <port>\n → OK\n),
    /// then returns a client ready for JSON-line commands.
    pub async fn connect(vsock_uds_path: &Path, timeout: Duration) -> Result<Self> {
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            if let Ok(stream) = UnixStream::connect(vsock_uds_path).await {
                // Firecracker vsock handshake
                let (read_half, mut write_half) = tokio::io::split(stream);
                let mut reader = BufReader::new(read_half);

                let cmd = format!("CONNECT {VSOCK_GUEST_PORT}\n");
                write_half.write_all(cmd.as_bytes()).await?;

                let mut response = String::new();
                reader.read_line(&mut response).await?;

                if response.starts_with("OK") {
                    return Ok(Self {
                        reader,
                        writer: write_half,
                    });
                }
            }

            if tokio::time::Instant::now() > deadline {
                bail!(
                    "Could not connect to guest via vsock at {:?}",
                    vsock_uds_path
                );
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Send a JSON command and read a JSON response (newline-delimited).
    async fn send_command(&mut self, cmd: &Value) -> Result<Value> {
        let msg = format!("{}\n", cmd);
        self.writer.write_all(msg.as_bytes()).await?;

        let mut line = String::new();
        let n = self
            .reader
            .read_line(&mut line)
            .await
            .context("read response from guest")?;
        if n == 0 {
            bail!("EOF from guest agent");
        }

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

    pub async fn reset(&mut self) -> Result<Vec<u8>> {
        let resp = self
            .send_command(&serde_json::json!({"cmd": "reset"}))
            .await?;
        decode_obs(&resp)
    }

    pub async fn step(&mut self, action: i64) -> Result<StepResult> {
        let resp = self
            .send_command(&serde_json::json!({"cmd": "step", "action": action}))
            .await?;
        Ok(StepResult {
            obs: decode_obs(&resp)?,
            reward: resp["reward"].as_f64().unwrap_or(0.0),
            done: resp["done"].as_bool().unwrap_or(false),
        })
    }

    pub async fn get_legal_actions(&mut self) -> Result<Vec<i64>> {
        let resp = self
            .send_command(&serde_json::json!({"cmd": "get_actions"}))
            .await?;
        let actions = resp["actions"]
            .as_array()
            .context("missing actions")?
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        Ok(actions)
    }
}

fn decode_obs(resp: &Value) -> Result<Vec<u8>> {
    let obs_b64 = resp["obs"].as_str().context("missing obs field")?;
    let obs = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, obs_b64)?;
    Ok(obs)
}

pub struct StepResult {
    pub obs: Vec<u8>,
    pub reward: f64,
    pub done: bool,
}
