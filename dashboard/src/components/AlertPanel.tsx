interface Alert {
  icao24: string;
  timestamp: number;
  alert_level: string;
  p_value: number;
  score: number;
  explanation: string;
}

interface AlertPanelProps {
  alerts: Alert[];
  onSelect: (icao: string) => void;
}

const LEVEL_STYLES: Record<string, { bg: string; text: string }> = {
  red: { bg: "#fef2f2", text: "#dc2626" },
  amber: { bg: "#fffbeb", text: "#d97706" },
  yellow: { bg: "#fefce8", text: "#ca8a04" },
};

function AlertPanel({ alerts, onSelect }: AlertPanelProps) {
  return (
    <div style={{ flex: 1, overflow: "auto", padding: "8px 0" }}>
      <h3 style={{ padding: "0 16px", fontSize: 14, color: "#666" }}>
        Alerts ({alerts.length})
      </h3>
      {alerts.length === 0 && (
        <p style={{ padding: "0 16px", fontSize: 13, color: "#999" }}>
          No alerts yet. Waiting for anomalies...
        </p>
      )}
      {alerts.map((alert, i) => {
        const style = LEVEL_STYLES[alert.alert_level] || LEVEL_STYLES.yellow;
        return (
          <div
            key={`${alert.icao24}-${alert.timestamp}-${i}`}
            onClick={() => onSelect(alert.icao24)}
            style={{
              padding: "8px 16px",
              borderBottom: "1px solid #f0f0f0",
              cursor: "pointer",
              background: style.bg,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span
                style={{
                  fontWeight: 600,
                  fontSize: 13,
                  color: style.text,
                  textTransform: "uppercase",
                }}
              >
                {alert.alert_level}
              </span>
              <span style={{ fontSize: 11, color: "#999" }}>
                {new Date(alert.timestamp * 1000).toLocaleTimeString()}
              </span>
            </div>
            <div style={{ fontSize: 13, marginTop: 2 }}>
              <strong>{alert.icao24}</strong> &mdash; p=
              {alert.p_value.toFixed(4)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default AlertPanel;
