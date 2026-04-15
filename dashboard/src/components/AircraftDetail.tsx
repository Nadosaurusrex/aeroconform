interface Aircraft {
  icao24: string;
  callsign: string | null;
  latitude: number;
  longitude: number;
  altitude: number;
  velocity: number;
  heading: number;
  vertical_rate: number;
  on_ground: boolean;
  anomaly_score: number;
  p_value: number;
  alert_level: string;
}

interface AircraftDetailProps {
  aircraft: Aircraft;
  onClose: () => void;
}

function AircraftDetail({ aircraft, onClose }: AircraftDetailProps) {
  return (
    <div
      style={{
        padding: 16,
        borderBottom: "1px solid #ddd",
        background: "#f9f9f9",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h3 style={{ margin: 0, fontSize: 15 }}>
          {aircraft.callsign || aircraft.icao24}
        </h3>
        <button
          onClick={onClose}
          style={{
            border: "none",
            background: "none",
            cursor: "pointer",
            fontSize: 16,
          }}
        >
          x
        </button>
      </div>
      <table style={{ fontSize: 12, marginTop: 8, width: "100%" }}>
        <tbody>
          <Row label="ICAO24" value={aircraft.icao24} />
          <Row label="Position" value={`${aircraft.latitude.toFixed(4)}, ${aircraft.longitude.toFixed(4)}`} />
          <Row label="Altitude" value={`${Math.round(aircraft.altitude)} m`} />
          <Row label="Speed" value={`${Math.round(aircraft.velocity)} m/s`} />
          <Row label="Heading" value={`${Math.round(aircraft.heading)}\u00B0`} />
          <Row label="Vert. Rate" value={`${aircraft.vertical_rate.toFixed(1)} m/s`} />
          <Row label="On Ground" value={aircraft.on_ground ? "Yes" : "No"} />
          <Row label="Alert" value={aircraft.alert_level.toUpperCase()} />
          <Row label="P-value" value={aircraft.p_value.toFixed(4)} />
          <Row label="Score" value={aircraft.anomaly_score.toFixed(2)} />
        </tbody>
      </table>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr>
      <td style={{ color: "#666", paddingRight: 12, paddingBottom: 2 }}>
        {label}
      </td>
      <td style={{ fontWeight: 500 }}>{value}</td>
    </tr>
  );
}

export default AircraftDetail;
