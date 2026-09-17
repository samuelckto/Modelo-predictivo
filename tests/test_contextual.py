"""Tests del modulo contextual: clima, lineup y pitcher changes.

Cada test usa datos simulados. No depende de ninguna API externa ni de la DB.
"""
from __future__ import annotations

import math

import pytest

from shared.contextual import (
    ROOF_CLOSED,
    VENUE_COORDS,
    _wind_component_to_cf,
    apply_adjustment,
    delta_lineup_change,
    delta_pitcher_change,
    delta_weather,
)


# ---------------------------------------------------------------------------
# Viento
# ---------------------------------------------------------------------------

class TestWindComponent:
    def test_wind_directly_to_cf(self):
        """Viento VA hacia CF (bearing 0 = Norte) -> positivo."""
        # Viento viene del Sur (180°), va al Norte (0°) = hacia CF si CF=0°
        comp = _wind_component_to_cf(20, 180, 0)
        assert comp > 0, f"esperaba positivo, obtuvo {comp}"

    def test_wind_directly_from_cf(self):
        """Viento viene DESDE CF -> negativo."""
        # Viento viene del Norte (0°), va al Sur (180°) = desde CF si CF=0°
        comp = _wind_component_to_cf(20, 0, 0)
        assert comp < 0, f"esperaba negativo, obtuvo {comp}"

    def test_crosswind_is_small(self):
        """Viento perpendicular -> componente cercano a 0."""
        comp = _wind_component_to_cf(20, 90, 0)
        assert abs(comp) < 2, f"esperaba ~0, obtuvo {comp}"


# ---------------------------------------------------------------------------
# Delta por clima
# ---------------------------------------------------------------------------

class TestDeltaWeather:
    def test_wrigley_wind_strong_to_cf(self):
        """Wrigley Field con viento fuerte hacia CF -> delta positivo en total."""
        wrigley_bearing = VENUE_COORDS["Wrigley Field"][2]  # 23°
        # Viento viene del sur-suroeste (203°) -> va al nor-noreste (23°)
        weather = {
            "wind_speed_kmh": 25, "wind_direction_deg": 203,
            "temperature_c": 28, "humidity": 50, "precipitation_mm": 0,
        }
        adj = delta_weather(weather, "Wrigley Field", "total")
        assert adj is not None
        assert adj["delta_pct"] > 0, f"esperaba delta positivo, obtuvo {adj}"

    def test_cold_day_reduces_total(self):
        """Dia frio -> delta negativo en total."""
        weather = {
            "wind_speed_kmh": 5, "wind_direction_deg": 0,
            "temperature_c": 5, "humidity": 50, "precipitation_mm": 0,
        }
        adj = delta_weather(weather, "Wrigley Field", "total")
        assert adj is not None
        assert adj["delta_pct"] < 0

    def test_hot_day_increases_total(self):
        """Dia caliente -> delta positivo en total."""
        weather = {
            "wind_speed_kmh": 5, "wind_direction_deg": 0,
            "temperature_c": 35, "humidity": 50, "precipitation_mm": 0,
        }
        adj = delta_weather(weather, "Wrigley Field", "total")
        assert adj is not None
        assert adj["delta_pct"] > 0

    def test_roof_closed_no_adjustment(self):
        """Estadio con techo cerrado -> sin ajuste."""
        weather = {
            "wind_speed_kmh": 30, "wind_direction_deg": 180,
            "temperature_c": 5, "humidity": 90, "precipitation_mm": 5,
        }
        adj = delta_weather(weather, "Tropicana Field", "total")
        assert adj is None

    def test_light_wind_no_adjustment(self):
        """Viento suave -> sin ajuste."""
        weather = {
            "wind_speed_kmh": 8, "wind_direction_deg": 0,
            "temperature_c": 22, "humidity": 50, "precipitation_mm": 0,
        }
        adj = delta_weather(weather, "Wrigley Field", "total")
        assert adj is None

    def test_moneyline_no_wind_adjustment(self):
        """El viento no ajusta moneyline, solo total."""
        weather = {
            "wind_speed_kmh": 30, "wind_direction_deg": 203,
            "temperature_c": 22, "humidity": 50, "precipitation_mm": 0,
        }
        adj = delta_weather(weather, "Wrigley Field", "moneyline")
        assert adj is None

    def test_all_venues_have_three_elements(self):
        """Todas las entradas de VENUE_COORDS tienen (lat, lon, bearing)."""
        for name, coords in VENUE_COORDS.items():
            assert len(coords) == 3, f"{name} tiene {len(coords)} elementos"

    def test_all_roof_venues_exist(self):
        """Todos los estadios con techo cerrado estan en VENUE_COORDS."""
        for name in ROOF_CLOSED:
            assert name in VENUE_COORDS, f"{name} no esta en VENUE_COORDS"


# ---------------------------------------------------------------------------
# Delta por cambio de pitcher
# ---------------------------------------------------------------------------

class TestDeltaPitcher:
    def test_worse_pitcher_negative_delta(self):
        """Pitcher nuevo con ERA peor -> delta negativo."""
        adj = delta_pitcher_change(expected_era=3.0, actual_era=5.5, is_home=True)
        assert adj is not None
        assert adj["delta_pct"] < 0

    def test_better_pitcher_positive_delta(self):
        """Pitcher nuevo con ERA mejor -> delta positivo."""
        adj = delta_pitcher_change(expected_era=5.0, actual_era=3.0, is_home=True)
        assert adj is not None
        assert adj["delta_pct"] > 0

    def test_similar_era_no_adjustment(self):
        """Diferencia de ERA < 1.0 -> sin ajuste."""
        adj = delta_pitcher_change(expected_era=3.5, actual_era=4.0, is_home=True)
        assert adj is None

    def test_none_era_no_adjustment(self):
        """ERA desconocida -> sin ajuste."""
        adj = delta_pitcher_change(expected_era=None, actual_era=4.0, is_home=True)
        assert adj is None

    def test_max_delta_capped(self):
        """El delta no puede pasar de ±5%."""
        adj = delta_pitcher_change(expected_era=2.0, actual_era=10.0, is_home=True)
        assert adj is not None
        assert abs(adj["delta_pct"]) <= 5.0


# ---------------------------------------------------------------------------
# Delta por cambio de lineup
# ---------------------------------------------------------------------------

class TestDeltaLineup:
    def test_elite_batter_missing(self):
        """Bateador con wRC+ 160 fuera -> ajuste negativo."""
        adj = delta_lineup_change(missing_wrc_plus=160, is_home=True)
        assert adj is not None
        assert adj["delta_pct"] < 0

    def test_average_batter_no_adjustment(self):
        """Bateador con wRC+ 100 fuera -> sin ajuste."""
        adj = delta_lineup_change(missing_wrc_plus=100, is_home=True)
        assert adj is None

    def test_none_no_adjustment(self):
        """Sin bateador faltante -> sin ajuste."""
        adj = delta_lineup_change(missing_wrc_plus=None, is_home=True)
        assert adj is None


# ---------------------------------------------------------------------------
# Aplicar ajuste a probabilidad
# ---------------------------------------------------------------------------

class TestApplyAdjustment:
    def test_basic_increase(self):
        """Sumar delta positivo sin lado."""
        p = apply_adjustment(0.60, 0.04, "total", True, None)
        assert abs(p - 0.64) < 0.001

    def test_clamp_at_99(self):
        """La probabilidad no puede pasar de 0.99."""
        p = apply_adjustment(0.97, 0.10, "total", True, None)
        assert p == 0.99

    def test_clamp_at_01(self):
        """La probabilidad no puede bajar de 0.01."""
        p = apply_adjustment(0.03, -0.10, "total", True, None)
        assert p == 0.01

    def test_side_matters(self):
        """Ajuste para el equipo visitante invierte el delta si el pick es local."""
        # Pick es home, ajuste es away negative -> invierte -> positivo para home
        p = apply_adjustment(0.60, -0.02, "moneyline", True, "away")
        assert p > 0.60

    def test_same_side_keeps_sign(self):
        """Ajuste para el equipo local se aplica directamente si pick es local."""
        p = apply_adjustment(0.60, -0.02, "moneyline", True, "home")
        assert p < 0.60
