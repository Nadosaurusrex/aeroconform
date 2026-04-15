"""Shared constants for AeroConform."""

from __future__ import annotations

# Model input dimension after sin/cos heading encoding
INPUT_DIM = 8

# Feature names in model input order
FEATURE_NAMES = (
    "latitude",
    "longitude",
    "baro_altitude",
    "velocity",
    "sin_track",
    "cos_track",
    "vertical_rate",
    "on_ground",
)

# LIMM FIR (Milan) bounding box
LIMM_BBOX_WEST = 6.5
LIMM_BBOX_SOUTH = 44.0
LIMM_BBOX_EAST = 13.5
LIMM_BBOX_NORTH = 47.0
LIMM_BBOX = (LIMM_BBOX_WEST, LIMM_BBOX_SOUTH, LIMM_BBOX_EAST, LIMM_BBOX_NORTH)

# OpenSky state_vectors_data4 columns we query
TRINO_COLUMNS = (
    "time",
    "icao24",
    "lat",
    "lon",
    "baroaltitude",
    "velocity",
    "heading",
    "vertrate",
    "onground",
    "callsign",
    "geoaltitude",
    "squawk",
)

# Flight segmentation
GAP_THRESHOLD_SECONDS = 30 * 60  # 30 minutes
MIN_FLIGHT_OBSERVATIONS = 20

# Context window for model
CONTEXT_LENGTH = 128

# OpenSky API
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
OPENSKY_API_BASE = "https://opensky-network.org/api"
OPENSKY_RATE_LIMIT_SECONDS = 5

# Earth radius for geodesic calculations (km)
EARTH_RADIUS_KM = 6371.0

# Unit conversions
NM_TO_KM = 1.852
FT_TO_M = 0.3048
