use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

const VSOCK_GUEST_PORT: u16 = 5005;

pub struct StepResult {
    pub obs: Vec<u8>,
    pub reward: f64,
    pub done: bool,
}

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
    /// After the Firecracker vsock handshake (`CONNECT` → `OK`), sends
    /// a lightweight `get_actions` probe to confirm the Python agent
    /// has actually `accept()`-ed this connection and is responsive.
    ///
    /// This matters after snapshot restore: the guest kernel needs to
    /// tear down the stale pre-snapshot vsock connection before Python
    /// cycles back to `accept()`. The Firecracker proxy returns `OK`
    /// immediately (it just queues the connect request), so without
    /// this probe we'd race the guest-side cleanup and occasionally
    /// get EOF when sending the real command.
    /// Attempt a single vsock connect + Firecracker handshake.
    /// Returns Err if the socket isn't available or the handshake fails.
    pub async fn connect(vsock_uds_path: &Path, timeout: Duration) -> Result<Self> {
        tokio::time::timeout(timeout, async {
            loop {
                if let Ok(mut client) = Self::try_connect_once(vsock_uds_path).await {
                    if client
                        .send_command(&serde_json::json!({"cmd": "get_actions"}))
                        .await
                        .is_ok()
                    {
                        return Ok(client);
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Could not connect to guest via vsock at {vsock_uds_path:?} within {timeout:?}"
            )
        })?
    }

    async fn try_connect_once(vsock_uds_path: &Path) -> Result<Self> {
        let stream = UnixStream::connect(vsock_uds_path).await?;
        let (read_half, mut write_half) = tokio::io::split(stream);
        let mut reader = BufReader::new(read_half);

        let cmd = format!("CONNECT {VSOCK_GUEST_PORT}\n");
        write_half.write_all(cmd.as_bytes()).await?;

        let mut response = String::new();
        reader.read_line(&mut response).await?;

        if response.starts_with("OK") {
            Ok(Self {
                reader,
                writer: write_half,
            })
        } else {
            bail!(
                "vsock handshake rejected: {response}",
                response = response.trim()
            );
        }
    }

    /// Send a JSON command and read a JSON response (newline-delimited).
    async fn send_command(&mut self, cmd: &Value) -> Result<Value> {
        self.writer.write_all(format!("{cmd}\n").as_bytes()).await?;

        let mut line = String::new();
        let bytes_read = self
            .reader
            .read_line(&mut line)
            .await
            .context("read response from guest")?;

        if bytes_read == 0 {
            bail!("EOF from guest agent");
        }

        let response: Value = serde_json::from_str(line.trim()).context("parse guest response")?;
        if response.get("status").and_then(|s| s.as_str()) == Some("error") {
            let msg = response
                .get("message")
                .and_then(|message| message.as_str())
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
        let response = self
            .send_command(&serde_json::json!({"cmd": "step", "action": action}))
            .await?;
        Ok(StepResult {
            obs: decode_obs(&response)?,
            reward: response["reward"].as_f64().unwrap_or(0.0),
            done: response["done"].as_bool().unwrap_or(false),
        })
    }

    pub async fn get_legal_actions(&mut self) -> Result<Vec<i64>> {
        let response = self
            .send_command(&serde_json::json!({"cmd": "get_actions"}))
            .await?;
        Ok(response["actions"]
            .as_array()
            .context("missing actions")?
            .iter()
            .filter_map(|v| v.as_i64())
            .collect())
    }
}

fn decode_obs(response: &Value) -> Result<Vec<u8>> {
    Ok({
        base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            response["obs"].as_str().context("missing obs field")?,
        )?
    })
}
