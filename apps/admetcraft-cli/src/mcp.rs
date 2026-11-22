use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use admet_calc::{MoleculeEvaluation, PkRoute, PkSimulationConfig, TppPreset};
use admet_engine::{FilterCriteria, MAX_READ_BYTES, PathSandbox};
use admet_qsar::{
    DEFAULT_MAX_METABOLITES, compare_smiles_auto, dg_kcal_to_ki_nm, evaluate_batch_auto,
    evaluate_smiles_auto, extract_mol_features, load_default_model_pack, load_default_sea_index,
    predict_metabolism_sites, run_metabolite_guard, simulate_human_pk_dose,
};
use admet_report::{ExportFormat, render_pk_profile_svg, write_report_to_writer};
use chem_core::{parse_csv_batch, parse_sdf, parse_smiles, parse_smiles_batch};
use serde::Serialize;
use serde_json::{Value, json};

/// Maximum incoming JSON-RPC request line size (1 MiB) to prevent memory amplification DoS.
pub const MAX_MCP_MESSAGE_BYTES: usize = 1024 * 1024;
/// Maximum number of molecules accepted in a single MCP tool call.
pub const MAX_MCP_MOLECULES: usize = 1_000;
/// Maximum concurrent TCP client worker threads to avoid resource exhaustion.
pub const MAX_TCP_CONNECTIONS: usize = 32;
/// Read/write socket timeout in TCP mode so idle sockets never starve the server.
pub const TCP_SOCKET_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

struct SessionAuth {
    initialized: bool,
    required_token: Option<String>,
}

/// Run the MCP server over `stdin`/`stdout` (no token required for local stdio pipe,
/// though `initialize` lifecycle is still enforced).
pub fn run_mcp_stdio(sandbox: &PathSandbox) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    serve_stream(reader, &mut writer, sandbox, None)
}

struct ConnGuard {
    active: Arc<AtomicUsize>,
}

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Run the MCP server over a TCP bridge address.
/// Enforces loopback-only binding unless `allow_remote` is true, and requires `auth_token`.
/// Spawns a bounded worker thread per connection with socket timeouts so idle clients cannot
/// block other callers.
pub fn run_mcp_tcp(
    addr_str: &str,
    sandbox: &PathSandbox,
    allow_remote: bool,
    auth_token: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = addr_str.parse()?;
    if !addr.ip().is_loopback() && !allow_remote {
        return Err(format!(
            "refusing to bind non-loopback TCP address {addr} without --allow-remote"
        )
        .into());
    }
    let token = auth_token
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .ok_or("TCP MCP mode requires a non-empty --mcp-token (or ADMETCRAFT_MCP_TOKEN)")?;

    let listener = TcpListener::bind(addr)?;
    eprintln!("AdmetCraft MCP TCP listening on {addr} (auth required)");
    let active_conns = Arc::new(AtomicUsize::new(0));
    let shared_sandbox = Arc::new(sandbox.clone());

    for stream_res in listener.incoming() {
        let Ok(stream) = stream_res else {
            continue;
        };
        handle_accepted_tcp_stream(
            stream,
            Arc::clone(&shared_sandbox),
            token.clone(),
            Arc::clone(&active_conns),
            TCP_SOCKET_TIMEOUT,
        );
    }
    Ok(())
}

fn handle_accepted_tcp_stream(
    mut stream: TcpStream,
    sandbox: Arc<PathSandbox>,
    token: String,
    active_conns: Arc<AtomicUsize>,
    timeout: Duration,
) {
    let prev = active_conns.fetch_add(1, Ordering::AcqRel);
    let guard = ConnGuard {
        active: Arc::clone(&active_conns),
    };
    if prev >= MAX_TCP_CONNECTIONS {
        let err_resp = JsonRpcResponse::err(
            Value::Null,
            -32001,
            "server busy: too many concurrent TCP connections",
        );
        let _ = write_json_line(&mut stream, &err_resp);
        drop(guard);
        return;
    }

    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));

    let Ok(reader_stream) = stream.try_clone() else {
        drop(guard);
        return;
    };

    std::thread::spawn(move || {
        let _guard = guard;
        let reader = BufReader::new(reader_stream);
        let mut writer = stream;
        let _ = serve_stream(reader, &mut writer, &sandbox, Some(token));
    });
}

#[cfg(test)]
pub fn serve_stream_for_test<R: BufRead, W: Write>(
    reader: R,
    writer: &mut W,
    sandbox: &PathSandbox,
    required_token: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    serve_stream(reader, writer, sandbox, required_token)
}

#[cfg(test)]
pub fn handle_tcp_stream_for_test(
    stream: TcpStream,
    sandbox: Arc<PathSandbox>,
    token: String,
    active_conns: Arc<AtomicUsize>,
    timeout: Duration,
) {
    handle_accepted_tcp_stream(stream, sandbox, token, active_conns, timeout);
}

#[cfg(test)]
pub fn run_isolated_for_test<F>(id: Value, f: F) -> Option<String>
where
    F: FnOnce() -> String,
{
    let resp = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(s) => JsonRpcResponse::ok(id, json!({ "ok": s })),
        Err(_) => JsonRpcResponse::err(
            id,
            -32603,
            "internal server error while handling MCP request",
        ),
    };
    serde_json::to_string(&resp).ok()
}

enum LineRead {
    Eof,
    Line(String),
    TooLarge,
}

fn read_limited_line<R: BufRead>(reader: &mut R, limit: usize) -> std::io::Result<LineRead> {
    let mut buf = Vec::new();
    let mut exceeded = false;
    loop {
        let avail = reader.fill_buf()?;
        if avail.is_empty() {
            if exceeded {
                return Ok(LineRead::TooLarge);
            }
            if buf.is_empty() {
                return Ok(LineRead::Eof);
            }
            return Ok(LineRead::Line(String::from_utf8_lossy(&buf).into_owned()));
        }
        if let Some(pos) = avail.iter().position(|&b| b == b'\n') {
            if !exceeded {
                if buf.len() + pos > limit {
                    exceeded = true;
                } else {
                    buf.extend_from_slice(&avail[..pos]);
                }
            }
            reader.consume(pos + 1);
            if exceeded {
                return Ok(LineRead::TooLarge);
            }
            return Ok(LineRead::Line(String::from_utf8_lossy(&buf).into_owned()));
        }
        let len = avail.len();
        if !exceeded {
            if buf.len() + len > limit {
                exceeded = true;
                buf.clear();
            } else {
                buf.extend_from_slice(avail);
            }
        }
        reader.consume(len);
    }
}

fn serve_stream<R: BufRead, W: Write>(
    mut reader: R,
    writer: &mut W,
    sandbox: &PathSandbox,
    required_token: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut auth = SessionAuth {
        initialized: false,
        required_token,
    };

    loop {
        match read_limited_line(&mut reader, MAX_MCP_MESSAGE_BYTES)? {
            LineRead::Eof => break,
            LineRead::TooLarge => {
                let err_resp = JsonRpcResponse::err(
                    Value::Null,
                    -32600,
                    format!("request exceeds maximum message size ({MAX_MCP_MESSAGE_BYTES} bytes)"),
                );
                write_json_line(writer, &err_resp)?;
                continue;
            }
            LineRead::Line(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let raw_val: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        let err_resp =
                            JsonRpcResponse::err(Value::Null, -32700, format!("parse error: {e}"));
                        write_json_line(writer, &err_resp)?;
                        continue;
                    }
                };

                let Some(obj) = raw_val.as_object() else {
                    let err_resp = JsonRpcResponse::err(
                        Value::Null,
                        -32600,
                        "invalid request: JSON-RPC message must be a JSON object",
                    );
                    write_json_line(writer, &err_resp)?;
                    continue;
                };

                // Distinguish absent "id" (notification) vs explicit "id": null (request).
                let parsed_id = match obj.get("id") {
                    None => None,
                    Some(v @ (Value::Null | Value::Number(_) | Value::String(_))) => {
                        Some(v.clone())
                    }
                    Some(_) => {
                        let err_resp = JsonRpcResponse::err(
                            Value::Null,
                            -32600,
                            "invalid request: 'id' must be a string, number, or null",
                        );
                        write_json_line(writer, &err_resp)?;
                        continue;
                    }
                };
                let resp_id = parsed_id.clone().unwrap_or(Value::Null);

                // Strictly validate JSON-RPC 2.0 version header.
                if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
                    let err_resp = JsonRpcResponse::err(
                        resp_id,
                        -32600,
                        "invalid request: 'jsonrpc' must be \"2.0\"",
                    );
                    write_json_line(writer, &err_resp)?;
                    continue;
                }

                let Some(method) = obj
                    .get("method")
                    .and_then(Value::as_str)
                    .filter(|m| !m.is_empty())
                else {
                    let err_resp = JsonRpcResponse::err(
                        resp_id,
                        -32600,
                        "invalid request: 'method' must be a non-empty string",
                    );
                    write_json_line(writer, &err_resp)?;
                    continue;
                };

                let params = match obj.get("params") {
                    None | Some(Value::Null) => json!({}),
                    Some(v @ Value::Object(_)) => v.clone(),
                    Some(_) => {
                        if parsed_id.is_some() {
                            let err_resp = JsonRpcResponse::err(
                                resp_id,
                                -32602,
                                "invalid params: 'params' must be a JSON object",
                            );
                            write_json_line(writer, &err_resp)?;
                        }
                        continue;
                    }
                };

                // JSON-RPC 2.0 §4.1: notifications (no 'id' member) must NOT receive any reply.
                let Some(id) = parsed_id else {
                    continue;
                };

                let resp = execute_request_isolated(id, method, &params, sandbox, &mut auth);
                write_json_line(writer, &resp)?;
            }
        }
    }
    Ok(())
}

fn execute_request_isolated(
    id: Value,
    method: &str,
    params: &Value,
    sandbox: &PathSandbox,
    auth: &mut SessionAuth,
) -> JsonRpcResponse {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle_request(id.clone(), method, params, sandbox, auth)
    })) {
        Ok(resp) => resp,
        Err(_) => JsonRpcResponse::err(
            id,
            -32603,
            "internal server error while handling MCP request",
        ),
    }
}

fn write_json_line<W: Write, T: Serialize>(
    writer: &mut W,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let s = serde_json::to_string(value)?;
    writer.write_all(s.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn handle_request(
    id: Value,
    method: &str,
    params: &Value,
    sandbox: &PathSandbox,
    auth: &mut SessionAuth,
) -> JsonRpcResponse {
    if method == "initialize" {
        if let Some(expected) = auth.required_token.as_deref() {
            let provided = params.get("token").and_then(Value::as_str).unwrap_or("");
            if !constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
                return JsonRpcResponse::err(
                    id,
                    -32001,
                    "unauthorized: invalid or missing MCP auth token",
                );
            }
        }
        auth.initialized = true;
        return JsonRpcResponse::ok(
            id,
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "admetcraft-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        );
    }

    if method == "ping" {
        return JsonRpcResponse::ok(id, json!({}));
    }

    if !auth.initialized {
        return JsonRpcResponse::err(
            id,
            -32002,
            "session not initialized; call 'initialize' first",
        );
    }

    match method {
        "tools/list" => JsonRpcResponse::ok(id, json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let Some(tool_name) = params
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            else {
                return JsonRpcResponse::err(
                    id,
                    -32602,
                    "invalid params: 'name' must be a non-empty string",
                );
            };
            let args = match params.get("arguments") {
                None | Some(Value::Null) => json!({}),
                Some(v @ Value::Object(_)) => v.clone(),
                Some(_) => {
                    return JsonRpcResponse::err(
                        id,
                        -32602,
                        "invalid params: 'arguments' must be a JSON object",
                    );
                }
            };
            match call_tool(tool_name, &args, sandbox) {
                Ok(result) => JsonRpcResponse::ok(id, result),
                Err(invalid_param_msg) => JsonRpcResponse::err(id, -32602, invalid_param_msg),
            }
        }
        other => JsonRpcResponse::err(id, -32601, format!("method not found: {other}")),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "evaluate_smiles",
            "description": "Evaluate up to 1000 SMILES strings (or a regular file <=10 MiB inside --read-root) for physicochemical, ionization/LogD7.4, lipophilicity, solubility, PK, druglikeness (QED, CNS MPO, Lipinski, Veber, Pfizer 3/75, GSK 4/400, Golden Triangle, Ro3), medchem (PAINS/Brenk/SA), and safety (hERG/AMES/DILI) metrics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "description": "Single SMILES string or array of SMILES strings (each 'SMILES [Name]').",
                        "oneOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "string" } }
                        ]
                    },
                    "file": {
                        "type": "string",
                        "description": "Optional path to .smi/.csv/.sdf regular file inside --read-root."
                    }
                }
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "count": { "type": "integer" },
                    "results": { "type": "array", "items": { "type": "object" } },
                    "errors": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["count", "results", "errors"]
            }
        },
        {
            "name": "filter_druglikeness",
            "description": "Filter up to 1000 SMILES compounds by Lipinski violations, QED, CNS MPO, BBB permeability, GI absorption, PAINS/Brenk alerts, and hERG/AMES/DILI safety flags.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of SMILES strings (each 'SMILES [Name]')."
                    },
                    "max_lipinski_violations": { "type": "integer", "minimum": 0 },
                    "bbb_permeant": { "type": "boolean" },
                    "gi_high": { "type": "boolean" },
                    "no_pains": { "type": "boolean" },
                    "no_brenk": { "type": "boolean" },
                    "min_qed": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "min_cns_mpo": { "type": "number", "minimum": 0.0, "maximum": 6.0 },
                    "no_high_herg": { "type": "boolean" },
                    "no_ames": { "type": "boolean" },
                    "no_dili": { "type": "boolean" }
                },
                "required": ["smiles"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "total_evaluated": { "type": "integer" },
                    "matched_count": { "type": "integer" },
                    "matched": { "type": "array", "items": { "type": "object" } },
                    "errors": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["total_evaluated", "matched_count", "matched", "errors"]
            }
        },
        {
            "name": "inspect_molecule",
            "description": "Return detailed structural breakdown (formula, exact mass, heavy atoms, rings, aromaticity, 2D coordinates), ionization, QED/CNS MPO, PAINS/Brenk, and safety alerts for a SMILES.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": { "type": "string" },
                    "name": { "type": "string" }
                },
                "required": ["smiles"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "canonical_smiles": { "type": "string" },
                    "formula": { "type": "string" },
                    "exact_mass": { "type": "number" },
                    "qed": { "type": "number" },
                    "cns_mpo": { "type": "number" },
                    "ionization": { "type": "object" },
                    "safety": { "type": "object" },
                    "warnings": { "type": "array", "items": { "type": "string" } },
                    "heavy_atoms": { "type": "integer" },
                    "aromatic_heavy_atoms": { "type": "integer" },
                    "rings": { "type": "array" },
                    "atoms": { "type": "array" },
                    "bonds": { "type": "array" },
                    "pains_alerts": { "type": "array", "items": { "type": "string" } },
                    "pains_catalog_size": { "type": "integer" },
                    "brenk_alerts": { "type": "array", "items": { "type": "string" } },
                    "brenk_catalog_size": { "type": "integer" },
                    "synthetic_accessibility": { "type": "number" }
                },
                "required": ["name", "canonical_smiles", "formula", "heavy_atoms", "pains_alerts", "brenk_alerts"]
            }
        },
        {
            "name": "compare_molecules",
            "description": "Compare two molecules (matched molecular pair / lead optimization delta) across MW, TPSA, WLOGP, cLogP, LogD7.4, ESOL LogS, QED, CNS MPO, SA score, and structural/safety alerts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles_a": { "type": "string", "description": "Reference/parent molecule SMILES." },
                    "smiles_b": { "type": "string", "description": "Candidate/analog molecule SMILES." },
                    "name_a": { "type": "string" },
                    "name_b": { "type": "string" }
                },
                "required": ["smiles_a", "smiles_b"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "name_a": { "type": "string" },
                    "name_b": { "type": "string" },
                    "delta_mw": { "type": "number" },
                    "delta_tpsa": { "type": "number" },
                    "delta_wlogp": { "type": "number" },
                    "delta_consensus_log_p": { "type": "number" },
                    "delta_logd_ph74": { "type": "number" },
                    "delta_esol_log_s": { "type": "number" },
                    "delta_qed": { "type": "number" },
                    "delta_cns_mpo": { "type": "number" },
                    "delta_sa_score": { "type": "number" },
                    "delta_pains": { "type": "integer" },
                    "delta_brenk": { "type": "integer" },
                    "eval_a": { "type": "object" },
                    "eval_b": { "type": "object" }
                },
                "required": ["name_a", "name_b", "delta_mw", "delta_tpsa", "delta_qed", "delta_cns_mpo", "eval_a", "eval_b"]
            }
        },
        {
            "name": "export_report",
            "description": "Atomically generate a CSV, JSON, FlateDecode PDF, or BOILED-Egg SVG report strictly inside --write-root using O_NOFOLLOW.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "output_path": { "type": "string" },
                    "format": {
                        "type": "string",
                        "enum": ["csv", "json", "pdf", "svg"]
                    }
                },
                "required": ["smiles", "output_path", "format"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "written_path": { "type": "string" },
                    "format": { "type": "string" },
                    "exported_count": { "type": "integer" },
                    "errors": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["written_path", "format", "exported_count", "errors"]
            }
        },
        {
            "name": "model_info",
            "description": "Return held-out Bemis-Murcko scaffold-split validation metrics and 90% conformal coverage for the installed local TDC QSAR model pack.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "loaded": { "type": "boolean" },
                    "version": { "type": "string" },
                    "endpoint_count": { "type": "integer" },
                    "endpoints": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["loaded", "endpoint_count"]
            }
        },
        {
            "name": "predict_metabolites",
            "description": "Predict Phase I & Phase II hepatic/plasma metabolites from SMILES and re-screen each metabolite for reactive bioactivation (epoxides, aniline N-OH, quinone-imines, acyl-glucuronides) and DILI/AMES/hERG toxicity escalation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": { "type": "string" },
                    "name": { "type": "string" },
                    "max_metabolites": { "type": "integer", "minimum": 1, "maximum": 24 }
                },
                "required": ["smiles"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "total_generated": { "type": "integer" },
                    "bioactivation_alerts": { "type": "integer" },
                    "escalation_count": { "type": "integer" },
                    "metabolites": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["total_generated", "bioactivation_alerts", "escalation_count", "metabolites"]
            }
        },
        {
            "name": "predict_offtargets",
            "description": "Predict human off-target safety pharmacology risks via pure-Rust SEA (Similarity Ensemble Approach) against the Bowes Safety-20 anti-target reference panel.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": { "type": "string" },
                    "name": { "type": "string" }
                },
                "required": ["smiles"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "panel_version": { "type": "string" },
                    "targets_evaluated": { "type": "integer" },
                    "high_risk_count": { "type": "integer" },
                    "moderate_risk_count": { "type": "integer" },
                    "hits": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["panel_version", "targets_evaluated", "high_risk_count", "hits"]
            }
        },
        {
            "name": "predict_human_pk_dose",
            "description": "Simulate mechanistic IVIVE -> Minimal-PBPK human pharmacokinetics and predicted QD/BID therapeutic oral/IV/SC doses with 90% conformal uncertainty intervals and optional SVG curve export inside --write-root.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "smiles": { "type": "string" },
                    "name": { "type": "string" },
                    "potency_nm": { "type": "number" },
                    "dg_kcal": { "type": "number" },
                    "target_fold": { "type": "number" },
                    "match_trough": { "type": "boolean" },
                    "route": { "type": "string", "enum": ["oral", "iv", "sc"] },
                    "out_svg_path": { "type": "string" }
                },
                "required": ["smiles"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "bcs_class": { "type": "string" },
                    "oral_bioavailability_f": { "type": "number" },
                    "t_half_ivive_h": { "type": "number" },
                    "qd": { "type": "object" },
                    "bid": { "type": "object" },
                    "recommended_regimen": { "type": "string" }
                },
                "required": ["bcs_class", "oral_bioavailability_f", "qd", "bid", "recommended_regimen"]
            }
        }
    ])
}

/// Build MCP success response without duplicating large JSON trees twice on the wire.
fn tool_ok(summary: String, payload: Value) -> Value {
    json!({
        "content": [{ "type": "text", "text": summary }],
        "structuredContent": payload,
        "isError": false
    })
}

fn tool_err(msg: impl Into<String>) -> Value {
    let text = msg.into();
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": true
    })
}

/// Dispatch tool call. Returns `Err(msg)` for JSON-RPC `-32602` Invalid params,
/// and `Ok(tool_ok/tool_err)` for valid protocol calls (including runtime/sandbox errors).
fn call_tool(name: &str, args: &Value, sandbox: &PathSandbox) -> Result<Value, String> {
    match name {
        "evaluate_smiles" => tool_evaluate_smiles(args, sandbox),
        "filter_druglikeness" => tool_filter_druglikeness(args),
        "inspect_molecule" => tool_inspect_molecule(args),
        "compare_molecules" => tool_compare_molecules(args),
        "export_report" => tool_export_report(args, sandbox),
        "model_info" => tool_model_info(),
        "predict_metabolites" => tool_predict_metabolites(args),
        "predict_offtargets" => tool_predict_offtargets(args),
        "predict_human_pk_dose" => tool_predict_human_pk_dose(args, sandbox),
        other => Err(format!("invalid params: unknown tool '{other}'")),
    }
}

fn parse_tpp_preset(args: &Value) -> TppPreset {
    match args
        .get("tpp_preset")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("cns") => TppPreset::CnsPenetrant,
        Some("peripheral") => TppPreset::PeripheralSelective,
        Some("oncology") => TppPreset::OncologyKinase,
        _ => TppPreset::OralSmallMolecule,
    }
}

fn tool_model_info() -> Result<Value, String> {
    if let Some(pack) = load_default_model_pack() {
        let eps: Vec<Value> = pack
            .models
            .iter()
            .map(|m| {
                json!({
                    "endpoint_id": m.endpoint_id,
                    "display_name": m.display_name,
                    "category": m.category,
                    "metrics": m.metrics,
                })
            })
            .collect();
        let summary = format!(
            "Loaded AdmetCraft TDC QSAR pack v{} ({} endpoints)",
            pack.version,
            pack.models.len()
        );
        Ok(tool_ok(
            summary,
            json!({
                "loaded": true,
                "version": pack.version,
                "split_method": pack.split_method,
                "endpoint_count": pack.models.len(),
                "endpoints": eps,
            }),
        ))
    } else {
        Ok(tool_ok(
            "No local TDC QSAR model pack installed yet (run `admetcraft-cli train-qsar`)."
                .to_string(),
            json!({
                "loaded": false,
                "endpoint_count": 0,
                "endpoints": [],
            }),
        ))
    }
}

enum SmilesInputError {
    InvalidParams(String),
    BatchLimitExceeded(String),
}

fn collect_smiles_inputs(args: &Value) -> Result<Vec<String>, SmilesInputError> {
    let Some(raw) = args.get("smiles") else {
        return Err(SmilesInputError::InvalidParams(
            "invalid params: missing required 'smiles'".to_string(),
        ));
    };
    let mut lines = Vec::new();
    if let Some(s) = raw.as_str() {
        for l in s.lines() {
            let t = l.trim();
            if !t.is_empty() {
                lines.push(t.to_string());
                if lines.len() > MAX_MCP_MOLECULES {
                    return Err(SmilesInputError::BatchLimitExceeded(format!(
                        "batch exceeds maximum allowed {MAX_MCP_MOLECULES} molecules per MCP call"
                    )));
                }
            }
        }
    } else if let Some(arr) = raw.as_array() {
        for (idx, item) in arr.iter().enumerate() {
            let Some(s) = item.as_str() else {
                return Err(SmilesInputError::InvalidParams(format!(
                    "invalid params: 'smiles[{idx}]' must be a string"
                )));
            };
            let t = s.trim();
            if !t.is_empty() {
                lines.push(t.to_string());
                if lines.len() > MAX_MCP_MOLECULES {
                    return Err(SmilesInputError::BatchLimitExceeded(format!(
                        "batch exceeds maximum allowed {MAX_MCP_MOLECULES} molecules per MCP call"
                    )));
                }
            }
        }
    } else {
        return Err(SmilesInputError::InvalidParams(
            "invalid params: 'smiles' must be a string or an array of strings".to_string(),
        ));
    }
    Ok(lines)
}

fn parse_optional_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(format!("invalid params: '{key}' must be a boolean")),
    }
}

fn parse_optional_non_neg_usize(args: &Value, key: &str) -> Result<Option<usize>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            if let Some(u) = v.as_u64() {
                let n = usize::try_from(u).map_err(|_| {
                    format!("invalid params: '{key}' exceeds platform integer range")
                })?;
                Ok(Some(n))
            } else {
                Err(format!(
                    "invalid params: '{key}' must be a non-negative integer"
                ))
            }
        }
    }
}

fn parse_optional_f64_range(
    args: &Value,
    key: &str,
    lo: f64,
    hi: f64,
) -> Result<Option<f64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let Some(n) = v.as_f64() else {
                return Err(format!("invalid params: '{key}' must be a number"));
            };
            if !n.is_finite() || n < lo || n > hi {
                return Err(format!(
                    "invalid params: '{key}' must be a finite number in [{lo}, {hi}]"
                ));
            }
            Ok(Some(n))
        }
    }
}

fn parse_optional_string<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(format!("invalid params: '{key}' must be a string")),
    }
}

fn eval_lines(lines: &[String], tpp: TppPreset) -> (Vec<MoleculeEvaluation>, Vec<Value>) {
    let joined = lines.join("\n");
    let parsed = parse_smiles_batch(&joined);
    let mut mols = Vec::new();
    let mut errors = Vec::new();
    for (line_no, raw, res) in parsed {
        match res {
            Ok(m) => mols.push(m),
            Err(e) => errors.push(json!({ "line": line_no, "input": raw, "error": e.to_string() })),
        }
    }
    (evaluate_batch_auto(&mols, tpp), errors)
}

fn tool_evaluate_smiles(args: &Value, sandbox: &PathSandbox) -> Result<Value, String> {
    let tpp = parse_tpp_preset(args);
    if let Some(file_val) = args.get("file")
        && !file_val.is_null()
    {
        let Some(file_str) = file_val.as_str().filter(|s| !s.trim().is_empty()) else {
            return Err("invalid params: 'file' must be a non-empty string".to_string());
        };
        let (resolved, content) =
            match sandbox.read_to_string_limited(Path::new(file_str), MAX_READ_BYTES) {
                Ok(res) => res,
                Err(e) => return Ok(tool_err(format!("read sandbox error: {e}"))),
            };
        let ext = resolved
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut mols = Vec::new();
        let mut errors = Vec::new();
        if ext == "sdf" || ext == "mol" {
            for (i, res) in parse_sdf(&content)
                .into_iter()
                .take(MAX_MCP_MOLECULES)
                .enumerate()
            {
                match res {
                    Ok(m) => mols.push(m),
                    Err(e) => errors.push(json!({ "record": i + 1, "error": e.to_string() })),
                }
            }
        } else if ext == "csv" {
            match parse_csv_batch(&content) {
                Ok(items) => {
                    for (i, res) in items.into_iter().take(MAX_MCP_MOLECULES).enumerate() {
                        match res {
                            Ok(m) => mols.push(m),
                            Err(e) => errors.push(json!({ "row": i + 2, "error": e.to_string() })),
                        }
                    }
                }
                Err(e) => return Ok(tool_err(format!("CSV parse error: {e}"))),
            }
        } else {
            for (line_no, raw, res) in parse_smiles_batch(&content)
                .into_iter()
                .take(MAX_MCP_MOLECULES)
            {
                match res {
                    Ok(m) => mols.push(m),
                    Err(e) => errors
                        .push(json!({ "line": line_no, "input": raw, "error": e.to_string() })),
                }
            }
        }
        let evals = evaluate_batch_auto(&mols, tpp);
        let summary = format!(
            "Evaluated {} molecule(s) from {}",
            evals.len(),
            resolved.display()
        );
        return Ok(tool_ok(
            summary,
            json!({
                "count": evals.len(),
                "results": evals,
                "errors": errors,
            }),
        ));
    }

    let lines = match collect_smiles_inputs(args) {
        Ok(l) => l,
        Err(SmilesInputError::InvalidParams(msg)) => return Err(msg),
        Err(SmilesInputError::BatchLimitExceeded(msg)) => return Ok(tool_err(msg)),
    };
    if lines.is_empty() {
        return Err(
            "invalid params: provide non-empty 'smiles' (string or array) or 'file' inside --read-root"
                .to_string(),
        );
    }
    let (evals, errors) = eval_lines(&lines, tpp);
    let summary = format!(
        "Evaluated {} molecule(s) ({} error(s))",
        evals.len(),
        errors.len()
    );
    Ok(tool_ok(
        summary,
        json!({
            "count": evals.len(),
            "results": evals,
            "errors": errors,
        }),
    ))
}

fn tool_filter_druglikeness(args: &Value) -> Result<Value, String> {
    let lines = match collect_smiles_inputs(args) {
        Ok(l) => l,
        Err(SmilesInputError::InvalidParams(msg)) => return Err(msg),
        Err(SmilesInputError::BatchLimitExceeded(msg)) => return Ok(tool_err(msg)),
    };
    if lines.is_empty() {
        return Err("invalid params: 'smiles' array must not be empty".to_string());
    }
    let criteria = FilterCriteria {
        max_lipinski_violations: parse_optional_non_neg_usize(args, "max_lipinski_violations")?,
        require_bbb: parse_optional_bool(args, "bbb_permeant")?,
        require_gi_high: parse_optional_bool(args, "gi_high")?,
        no_pains: parse_optional_bool(args, "no_pains")?.unwrap_or(false),
        no_brenk: parse_optional_bool(args, "no_brenk")?.unwrap_or(false),
        min_qed: parse_optional_f64_range(args, "min_qed", 0.0, 1.0)?,
        min_cns_mpo: parse_optional_f64_range(args, "min_cns_mpo", 0.0, 6.0)?,
        no_high_herg: parse_optional_bool(args, "no_high_herg")?.unwrap_or(false),
        no_ames: parse_optional_bool(args, "no_ames")?.unwrap_or(false),
        no_dili: parse_optional_bool(args, "no_dili")?.unwrap_or(false),
    };
    let (evals, errors) = eval_lines(&lines, parse_tpp_preset(args));
    let total = evals.len();
    let passed: Vec<MoleculeEvaluation> =
        evals.into_iter().filter(|e| criteria.matches(e)).collect();
    let summary = format!("Matched {}/{} molecule(s)", passed.len(), total);
    Ok(tool_ok(
        summary,
        json!({
            "total_evaluated": total,
            "matched_count": passed.len(),
            "matched": passed,
            "errors": errors,
        }),
    ))
}

fn tool_inspect_molecule(args: &Value) -> Result<Value, String> {
    let Some(smi) = args
        .get("smiles")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'smiles'".to_string());
    };
    let name = parse_optional_string(args, "name")?;
    let mut mol = match parse_smiles(smi) {
        Ok(m) => m,
        Err(e) => return Ok(tool_err(format!("invalid SMILES: {e}"))),
    };
    chem_core::generate_2d_coords(&mut mol);
    let eval = match evaluate_smiles_auto(smi, name, parse_tpp_preset(args)) {
        Ok(e) => e,
        Err(e) => return Ok(tool_err(format!("evaluation failed: {e}"))),
    };
    let summary = format!(
        "Inspected {} ({}, QED={:.3}, CNS MPO={:.2})",
        eval.name, eval.physicochemical.formula, eval.druglikeness.qed, eval.druglikeness.cns_mpo
    );
    Ok(tool_ok(
        summary,
        json!({
            "name": eval.name,
            "canonical_smiles": eval.canonical_smiles,
            "formula": eval.physicochemical.formula,
            "exact_mass": eval.physicochemical.exact_mass,
            "qed": eval.druglikeness.qed,
            "cns_mpo": eval.druglikeness.cns_mpo,
            "ionization": eval.ionization,
            "safety": eval.safety,
            "qsar": eval.qsar,
            "metabolism_sites": eval.metabolism_sites,
            "tpp_mpo": eval.tpp_mpo,
            "tox21_summary": eval.tox21_summary,
            "off_targets": eval.off_targets,
            "metabolite_guard": eval.metabolite_guard,
            "warnings": eval.warnings,
            "heavy_atoms": mol.heavy_atom_count(),
            "aromatic_heavy_atoms": mol.aromatic_heavy_atom_count(),
            "rings": mol.rings,
            "atoms": mol.atoms,
            "bonds": mol.bonds,
            "pains_alerts": eval.medicinal_chemistry.pains_alerts,
            "pains_catalog_size": eval.medicinal_chemistry.pains_catalog_size,
            "brenk_alerts": eval.medicinal_chemistry.brenk_alerts,
            "brenk_catalog_size": eval.medicinal_chemistry.brenk_catalog_size,
            "synthetic_accessibility": eval.medicinal_chemistry.synthetic_accessibility,
        }),
    ))
}

fn tool_compare_molecules(args: &Value) -> Result<Value, String> {
    let Some(smi_a) = args
        .get("smiles_a")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'smiles_a'".to_string());
    };
    let Some(smi_b) = args
        .get("smiles_b")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'smiles_b'".to_string());
    };
    let name_a = parse_optional_string(args, "name_a")?;
    let name_b = parse_optional_string(args, "name_b")?;
    let cmp = match compare_smiles_auto(smi_a, name_a, smi_b, name_b, parse_tpp_preset(args)) {
        Ok(c) => c,
        Err(e) => return Ok(tool_err(format!("comparison failed: {e}"))),
    };
    let summary = format!(
        "Compared {} vs {}: dMW={:+.2}, dTPSA={:+.2}, dLogD7.4={:+.2}, dQED={:+.3}, dCNS_MPO={:+.2}",
        cmp.name_a,
        cmp.name_b,
        cmp.delta_mw,
        cmp.delta_tpsa,
        cmp.delta_logd_ph74,
        cmp.delta_qed,
        cmp.delta_cns_mpo
    );
    let payload = serde_json::to_value(&cmp).unwrap_or_else(|_| json!({}));
    Ok(tool_ok(summary, payload))
}

fn tool_export_report(args: &Value, sandbox: &PathSandbox) -> Result<Value, String> {
    let lines = match collect_smiles_inputs(args) {
        Ok(l) => l,
        Err(SmilesInputError::InvalidParams(msg)) => return Err(msg),
        Err(SmilesInputError::BatchLimitExceeded(msg)) => return Ok(tool_err(msg)),
    };
    if lines.is_empty() {
        return Err("invalid params: 'smiles' list must not be empty".to_string());
    }
    let Some(out_str) = args
        .get("output_path")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'output_path'".to_string());
    };
    // Strictly require 'format' with no silent fallback to 'csv'.
    let Some(fmt_raw) = args.get("format").and_then(Value::as_str) else {
        return Err(
            "invalid params: missing required string 'format' (expected 'csv', 'json', 'pdf', or 'svg')"
                .to_string(),
        );
    };
    let fmt_str = fmt_raw.trim().to_ascii_lowercase();
    let format = match fmt_str.as_str() {
        "csv" => ExportFormat::Csv,
        "json" => ExportFormat::Json,
        "pdf" => ExportFormat::Pdf,
        "svg" => ExportFormat::SvgBoiledEgg,
        other => {
            return Err(format!(
                "invalid params: unsupported format '{other}' (expected 'csv', 'json', 'pdf', or 'svg')"
            ));
        }
    };

    let (evals, errors) = eval_lines(&lines, parse_tpp_preset(args));
    if evals.is_empty() {
        return Ok(tool_err("no valid molecules to export"));
    }

    // Atomically validate and open with O_NOFOLLOW, writing directly to the fd (no TOCTOU)
    let (resolved, mut file) = match sandbox.open_write(Path::new(out_str)) {
        Ok(pair) => pair,
        Err(e) => return Ok(tool_err(format!("write sandbox error: {e}"))),
    };
    if let Err(e) = write_report_to_writer(&evals, &mut file, format) {
        return Ok(tool_err(format!("export error: {e}")));
    }

    let summary = format!(
        "Exported {} molecule(s) to {}",
        evals.len(),
        resolved.display()
    );
    Ok(tool_ok(
        summary,
        json!({
            "written_path": resolved.display().to_string(),
            "format": fmt_str,
            "exported_count": evals.len(),
            "errors": errors,
        }),
    ))
}

fn tool_predict_metabolites(args: &Value) -> Result<Value, String> {
    let Some(smi) = args
        .get("smiles")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'smiles'".to_string());
    };
    let name = parse_optional_string(args, "name")?;
    let max_met =
        parse_optional_non_neg_usize(args, "max_metabolites")?.unwrap_or(DEFAULT_MAX_METABOLITES);
    let mut mol = match parse_smiles(smi) {
        Ok(m) => m,
        Err(e) => return Ok(tool_err(format!("invalid SMILES: {e}"))),
    };
    if let Some(n) = name {
        mol.name = Some(n.to_string());
    }
    let pack = load_default_model_pack();
    let mut parent_eval = admet_calc::evaluate_molecule(&mol);
    if let Some(mp) = pack {
        let feat = extract_mol_features(&mol, &parent_eval);
        parent_eval.qsar = Some(mp.predict_all(&feat, "mcp"));
    }
    let sites = predict_metabolism_sites(&mol);
    let report = run_metabolite_guard(&mol, &parent_eval, &sites, pack, max_met);
    let summary = format!(
        "Generated {} metabolite(s) for {} ({} bioactivation, {} escalation)",
        report.total_generated,
        parent_eval.name,
        report.bioactivation_alerts,
        report.escalation_count
    );
    let payload = serde_json::to_value(&report).unwrap_or_else(|_| json!({}));
    Ok(tool_ok(summary, payload))
}

fn tool_predict_offtargets(args: &Value) -> Result<Value, String> {
    let Some(smi) = args
        .get("smiles")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'smiles'".to_string());
    };
    let name = parse_optional_string(args, "name")?;
    let mut mol = match parse_smiles(smi) {
        Ok(m) => m,
        Err(e) => return Ok(tool_err(format!("invalid SMILES: {e}"))),
    };
    if let Some(n) = name {
        mol.name = Some(n.to_string());
    }
    let report = load_default_sea_index().predict_molecule(&mol);
    let summary = format!(
        "SEA off-target scan for {}: {} High-risk, {} Moderate-risk across {} targets",
        mol.name.as_deref().unwrap_or("Query"),
        report.high_risk_count,
        report.moderate_risk_count,
        report.targets_evaluated
    );
    let payload = serde_json::to_value(&report).unwrap_or_else(|_| json!({}));
    Ok(tool_ok(summary, payload))
}

fn tool_predict_human_pk_dose(args: &Value, sandbox: &PathSandbox) -> Result<Value, String> {
    let Some(smi) = args
        .get("smiles")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Err("invalid params: missing required non-empty string 'smiles'".to_string());
    };
    let name = parse_optional_string(args, "name")?;
    let potency_nm = args
        .get("dg_kcal")
        .and_then(Value::as_f64)
        .map(dg_kcal_to_ki_nm)
        .unwrap_or_else(|| {
            args.get("potency_nm")
                .and_then(Value::as_f64)
                .unwrap_or(10.0)
                .max(0.01)
        });
    let target_fold = args
        .get("target_fold")
        .and_then(Value::as_f64)
        .unwrap_or(1.0)
        .clamp(0.1, 100.0);
    let match_trough = args
        .get("match_trough")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let route = match args.get("route").and_then(Value::as_str) {
        Some("iv") => PkRoute::Iv,
        Some("sc") => PkRoute::Sc,
        _ => PkRoute::Oral,
    };
    let eval = match evaluate_smiles_auto(smi, name, parse_tpp_preset(args)) {
        Ok(e) => e,
        Err(e) => return Ok(tool_err(format!("invalid SMILES: {e}"))),
    };
    let cfg = PkSimulationConfig {
        target_potency_nm: potency_nm,
        target_fold,
        match_trough,
        route,
        ..Default::default()
    };
    let report = simulate_human_pk_dose(&eval, &cfg);
    if let Some(svg_out) = parse_optional_string(args, "out_svg_path")? {
        let (_, mut file) = match sandbox.open_write(Path::new(svg_out)) {
            Ok(pair) => pair,
            Err(e) => return Ok(tool_err(format!("write sandbox error: {e}"))),
        };
        if let Err(e) = file.write_all(render_pk_profile_svg(&report).as_bytes()) {
            return Ok(tool_err(format!("svg write error: {e}")));
        }
    }
    let summary = format!(
        "Human PK/PBPK for {}: {} | F={:.1}% | t1/2={:.1}h | {}",
        report.molecule_name,
        report.recommended_regimen,
        report.oral_bioavailability_f * 100.0,
        report.t_half_ivive_h,
        report.bcs_class.short_label()
    );
    let payload = serde_json::to_value(&report).unwrap_or_else(|_| json!({}));
    Ok(tool_ok(summary, payload))
}
