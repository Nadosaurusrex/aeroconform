import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

interface Aircraft {
  icao24: string;
  callsign: string | null;
  latitude: number;
  longitude: number;
  altitude: number;
  velocity: number;
  heading: number;
  alert_level: string;
  p_value: number;
}

interface MapProps {
  aircraft: Aircraft[];
  selectedIcao: string | null;
  onSelect: (icao: string) => void;
}

const ALERT_COLORS: Record<string, string> = {
  red: "#ef4444",
  amber: "#f59e0b",
  yellow: "#eab308",
  normal: "#22c55e",
};

// LIMM FIR center
const CENTER: [number, number] = [45.5, 10.0];

function Map({ aircraft, selectedIcao, onSelect }: MapProps) {
  return (
    <MapContainer
      center={CENTER}
      zoom={7}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://carto.com">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      {aircraft.map((ac) => (
        <CircleMarker
          key={ac.icao24}
          center={[ac.latitude, ac.longitude]}
          radius={ac.icao24 === selectedIcao ? 8 : 5}
          fillColor={ALERT_COLORS[ac.alert_level] || ALERT_COLORS.normal}
          fillOpacity={0.9}
          color={ac.icao24 === selectedIcao ? "white" : "transparent"}
          weight={ac.icao24 === selectedIcao ? 2 : 0}
          eventHandlers={{
            click: () => onSelect(ac.icao24),
          }}
        >
          <Popup>
            <div style={{ fontSize: 12 }}>
              <strong>{ac.callsign || ac.icao24}</strong>
              <br />
              Alt: {Math.round(ac.altitude)}m | Spd: {Math.round(ac.velocity)}m/s
              <br />
              Alert: {ac.alert_level} (p={ac.p_value.toFixed(4)})
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}

export default Map;
