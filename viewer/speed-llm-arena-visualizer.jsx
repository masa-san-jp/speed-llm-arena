// speed-llm-arena-visualizer.jsx
//
// results.json を読み取り専用で表示するビューアMVP(Issue #3)。
// バックエンド・認証・結果ファイルの書き換え・event_log再生は対象外。
// JSON不正・schema_version不一致・必須フィールド欠損は黙って空表示にせず、
// 明示的なエラーとして画面に表示する。
//
// 単体のReactコンポーネントとして、既存のReactアプリに組み込んで使う。
// 依存はReactのみ(useState)。

import React, { useState } from "react";

const SUPPORTED_SCHEMA_VERSION = 1;

const REQUIRED_TOP_FIELDS = ["schema_version", "config", "ranking", "matches"];
const REQUIRED_CONFIG_FIELDS = [
  "mode", "models", "hosts", "games_per_pair", "seed",
  "max_duration", "protocol_version", "max_tokens", "temperature",
];
const REQUIRED_RANKING_FIELDS = [
  "name", "elo", "win", "loss", "draw", "parse_error_rate", "ranking_valid",
];
const REQUIRED_MATCH_FIELDS = [
  "seed", "p0", "p1", "winner", "reason", "duration", "flips", "valid", "stats",
];
const REQUIRED_STAT_FIELDS = [
  "agent", "calls", "plays", "invalid_moves", "parse_errors", "api_errors",
  "think_time", "avg_latency", "warmup_status", "warmup_duration",
];

function missingKeys(obj, keys) {
  return keys.filter((k) => !(k in obj));
}

/**
 * results.json の契約(docs/arena-spec.md)に対して検証する。
 * 検証に失敗した場合は理由を含むエラー文字列を返す。成功時はnull。
 */
function validateResults(data) {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return "results.json のルートはオブジェクトである必要があります。";
  }

  const missingTop = missingKeys(data, REQUIRED_TOP_FIELDS);
  if (missingTop.length > 0) {
    return `必須フィールドが不足しています: ${missingTop.join(", ")}`;
  }

  if (data.schema_version !== SUPPORTED_SCHEMA_VERSION) {
    return `未対応の schema_version です: ${JSON.stringify(data.schema_version)} `
      + `(対応バージョン: ${SUPPORTED_SCHEMA_VERSION})`;
  }

  if (typeof data.config !== "object" || data.config === null) {
    return "config はオブジェクトである必要があります。";
  }
  const missingConfig = missingKeys(data.config, REQUIRED_CONFIG_FIELDS);
  if (missingConfig.length > 0) {
    return `config の必須フィールドが不足しています: ${missingConfig.join(", ")}`;
  }

  if (!Array.isArray(data.ranking)) {
    return "ranking は配列である必要があります。";
  }
  for (let i = 0; i < data.ranking.length; i++) {
    const missing = missingKeys(data.ranking[i] ?? {}, REQUIRED_RANKING_FIELDS);
    if (missing.length > 0) {
      return `ranking[${i}] の必須フィールドが不足しています: ${missing.join(", ")}`;
    }
  }

  if (!Array.isArray(data.matches)) {
    return "matches は配列である必要があります。";
  }
  for (let i = 0; i < data.matches.length; i++) {
    const m = data.matches[i] ?? {};
    const missing = missingKeys(m, REQUIRED_MATCH_FIELDS);
    if (missing.length > 0) {
      return `matches[${i}] の必須フィールドが不足しています: ${missing.join(", ")}`;
    }
    if (!Array.isArray(m.stats)) {
      return `matches[${i}].stats は配列である必要があります。`;
    }
    for (let j = 0; j < m.stats.length; j++) {
      const missingStat = missingKeys(m.stats[j] ?? {}, REQUIRED_STAT_FIELDS);
      if (missingStat.length > 0) {
        return `matches[${i}].stats[${j}] の必須フィールドが不足しています: ${missingStat.join(", ")}`;
      }
    }
  }

  return null;
}

const styles = {
  page: {
    fontFamily: "system-ui, -apple-system, sans-serif",
    color: "#1a1a1a",
    maxWidth: 1000,
    margin: "0 auto",
    padding: 24,
  },
  h1: { fontSize: 22, marginBottom: 4 },
  sub: { color: "#666", fontSize: 13, marginBottom: 20 },
  error: {
    background: "#fdecea",
    border: "1px solid #f5c2c0",
    color: "#8a1f14",
    padding: "12px 16px",
    borderRadius: 8,
    fontSize: 14,
    whiteSpace: "pre-wrap",
  },
  section: { marginTop: 28 },
  h2: { fontSize: 16, marginBottom: 10 },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: {
    textAlign: "left", borderBottom: "2px solid #ddd", padding: "6px 8px",
    color: "#555", fontWeight: 600,
  },
  td: { borderBottom: "1px solid #eee", padding: "6px 8px" },
  invalidRow: { background: "#fff6e5" },
  badge: {
    display: "inline-block", padding: "1px 8px", borderRadius: 10,
    fontSize: 11, fontWeight: 600,
  },
  badgeOk: { background: "#e6f4ea", color: "#1e7a34" },
  badgeWarn: { background: "#fdecea", color: "#8a1f14" },
  configGrid: {
    display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
    gap: 8, fontSize: 13,
  },
  configItem: { background: "#f7f7f8", borderRadius: 6, padding: "6px 10px" },
  matchCard: {
    border: "1px solid #e5e5e5", borderRadius: 8, padding: 12, marginBottom: 10,
  },
  matchHeader: { display: "flex", justifyContent: "space-between", fontSize: 14 },
  statTable: { width: "100%", borderCollapse: "collapse", fontSize: 12, marginTop: 8 },
};

function ConfigPanel({ config }) {
  const entries = [
    ["mode", config.mode],
    ["protocol_version", config.protocol_version],
    ["models", (config.models || []).join(", ")],
    ["hosts", (config.hosts || []).join(", ") || "-"],
    ["games_per_pair", config.games_per_pair],
    ["seed", config.seed],
    ["max_duration", `${config.max_duration}s`],
    ["max_tokens", config.max_tokens],
    ["temperature", config.temperature],
  ];
  return (
    <div style={styles.configGrid}>
      {entries.map(([k, v]) => (
        <div key={k} style={styles.configItem}>
          <div style={{ color: "#888", fontSize: 11 }}>{k}</div>
          <div>{String(v)}</div>
        </div>
      ))}
    </div>
  );
}

function RankingTable({ ranking }) {
  return (
    <table style={styles.table}>
      <thead>
        <tr>
          <th style={styles.th}>#</th>
          <th style={styles.th}>Model</th>
          <th style={styles.th}>Elo</th>
          <th style={styles.th}>W</th>
          <th style={styles.th}>L</th>
          <th style={styles.th}>D</th>
          <th style={styles.th}>Parse Error Rate</th>
          <th style={styles.th}>Status</th>
        </tr>
      </thead>
      <tbody>
        {ranking.map((r, i) => (
          <tr key={r.name} style={r.ranking_valid ? undefined : styles.invalidRow}>
            <td style={styles.td}>{i + 1}</td>
            <td style={styles.td}>{r.name}</td>
            <td style={styles.td}>{r.elo}</td>
            <td style={styles.td}>{r.win}</td>
            <td style={styles.td}>{r.loss}</td>
            <td style={styles.td}>{r.draw}</td>
            <td style={styles.td}>{(r.parse_error_rate * 100).toFixed(2)}%</td>
            <td style={styles.td}>
              {r.ranking_valid ? (
                <span style={{ ...styles.badge, ...styles.badgeOk }}>valid</span>
              ) : (
                <span style={{ ...styles.badge, ...styles.badgeWarn }} title="parse_error_rate > 1% (10+ games)">
                  invalid
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MatchStatsTable({ stats }) {
  return (
    <table style={styles.statTable}>
      <thead>
        <tr>
          <th style={styles.th}>Agent</th>
          <th style={styles.th}>Calls</th>
          <th style={styles.th}>Plays</th>
          <th style={styles.th}>Invalid</th>
          <th style={styles.th}>Parse Err</th>
          <th style={styles.th}>API Err</th>
          <th style={styles.th}>Avg Latency</th>
          <th style={styles.th}>Warmup</th>
        </tr>
      </thead>
      <tbody>
        {stats.map((s) => (
          <tr key={s.agent}>
            <td style={styles.td}>{s.agent}</td>
            <td style={styles.td}>{s.calls}</td>
            <td style={styles.td}>{s.plays}</td>
            <td style={styles.td}>{s.invalid_moves}</td>
            <td style={styles.td}>{s.parse_errors}</td>
            <td style={styles.td}>{s.api_errors}</td>
            <td style={styles.td}>{s.avg_latency}s</td>
            <td style={styles.td}>
              {s.warmup_status} ({s.warmup_duration}s)
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MatchList({ matches }) {
  return (
    <div>
      {matches.map((m, i) => (
        <div key={i} style={styles.matchCard}>
          <div style={styles.matchHeader}>
            <div>
              <strong>{m.p0}</strong> vs <strong>{m.p1}</strong>
              {!m.valid && (
                <span style={{ ...styles.badge, ...styles.badgeWarn, marginLeft: 8 }}>
                  invalid match
                </span>
              )}
            </div>
            <div style={{ color: "#666" }}>
              seed={m.seed} · {m.duration}s · flips={m.flips}
            </div>
          </div>
          <div style={{ fontSize: 13, marginTop: 4 }}>
            Winner: <strong>{m.winner ?? "draw / n/a"}</strong> ({m.reason})
          </div>
          <MatchStatsTable stats={m.stats} />
        </div>
      ))}
    </div>
  );
}

export default function SpeedArenaVisualizer() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);

  function loadFromText(text, name) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      setData(null);
      setError(`JSONとして解釈できません: ${e.message}`);
      setFileName(name);
      return;
    }
    const validationError = validateResults(parsed);
    if (validationError) {
      setData(null);
      setError(validationError);
      setFileName(name);
      return;
    }
    setData(parsed);
    setError(null);
    setFileName(name);
  }

  function handleFileChange(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadFromText(String(reader.result), file.name);
    reader.onerror = () => {
      setData(null);
      setError(`ファイルの読み込みに失敗しました: ${reader.error}`);
    };
    reader.readAsText(file);
  }

  return (
    <div style={styles.page}>
      <h1 style={styles.h1}>speed-llm-arena viewer</h1>
      <div style={styles.sub}>results.json を読み取り専用で表示します(書き込み・再実行はできません)。</div>

      <input type="file" accept="application/json,.json" onChange={handleFileChange} />
      {fileName && <span style={{ marginLeft: 8, color: "#666", fontSize: 13 }}>{fileName}</span>}

      {error && (
        <div style={{ ...styles.section }}>
          <div style={styles.error}>{error}</div>
        </div>
      )}

      {data && !error && (
        <>
          <div style={styles.section}>
            <h2 style={styles.h2}>Config</h2>
            <ConfigPanel config={data.config} />
          </div>

          <div style={styles.section}>
            <h2 style={styles.h2}>Ranking</h2>
            <RankingTable ranking={data.ranking} />
          </div>

          <div style={styles.section}>
            <h2 style={styles.h2}>Matches ({data.matches.length})</h2>
            <MatchList matches={data.matches} />
          </div>
        </>
      )}
    </div>
  );
}
