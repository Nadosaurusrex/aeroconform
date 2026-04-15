import { useEffect, useRef, useState } from "react";
import AlertPanel from "./components/AlertPanel";
import AircraftDetail from "./components/AircraftDetail";
import Map from "./components/Map";

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

interface Alert {
  icao24: string;
  timestamp: number;
  alert_level: string;
  p_value: number;
  score: number;
  latitude: number;
  longitude: number;
  altitude: number;
  explanation: string;
}

function App() {
  const [aircraft, setAircraft] = useState<Aircraft[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAircraft, setSelectedAircraft] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Poll aircraft state
  useEffect(() => {
    const fetchAircraft = async () => {
      try {
        const res = await fetch("/api/aircraft");
        if (res.ok) {
          setAircraft(await res.json());
        }
      } catch {
        // API not available yet
      }
    };

    fetchAircraft();
    const interval = setInterval(fetchAircraft, 5000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket for live alerts
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(
        `ws://${window.location.hostname}:8000/ws/alerts`
      );
      ws.onmessage = (event) => {
        const alert: Alert = JSON.parse(event.data);
        setAlerts((prev) => [alert, ...prev].slice(0, 100));
      };
      ws.onclose = () => {
        setTimeout(connect, 5000);
      };
      wsRef.current = ws;
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const selected = aircraft.find((a) => a.icao24 === selectedAircraft);

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "system-ui" }}>
      <div style={{ flex: 1, position: "relative" }}>
        <Map
          aircraft={aircraft}
          selectedIcao={selectedAircraft}
          onSelect={setSelectedAircraft}
        />
      </div>
      <div
        style={{
          width: 380,
          borderLeft: "1px solid #ddd",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid #ddd",
            background: "#1a1a2e",
            color: "white",
          }}
        >
          <h1 style={{ margin: 0, fontSize: 18 }}>AeroConform</h1>
          <p style={{ margin: "4px 0 0", fontSize: 12, opacity: 0.7 }}>
            {aircraft.length} aircraft tracked
          </p>
        </div>
        {selected && (
          <AircraftDetail
            aircraft={selected}
            onClose={() => setSelectedAircraft(null)}
          />
        )}
        <AlertPanel alerts={alerts} onSelect={setSelectedAircraft} />
      </div>
    </div>
  );
}

export default App;
