"""Tests for synthetic anomaly injection."""

from __future__ import annotations

import numpy as np
import pytest

from aeroconform.data.synthetic_anomalies import AnomalyInjector, generate_evaluation_set


@pytest.fixture
def injector() -> AnomalyInjector:
    """Create an anomaly injector with a fixed seed."""
    return AnomalyInjector(rng=np.random.default_rng(42))


@pytest.fixture
def clean_traj() -> np.ndarray:
    """Generate a clean reference trajectory."""
    t = 200
    traj = np.zeros((t, 6))
    traj[:, 0] = np.linspace(45.0, 46.0, t)  # lat
    traj[:, 1] = np.linspace(9.0, 10.0, t)   # lon
    traj[:, 2] = np.full(t, 35000.0)          # alt
    traj[:, 3] = np.full(t, 450.0)            # vel
    traj[:, 4] = np.full(t, 45.0)             # hdg
    traj[:, 5] = np.zeros(t)                  # vrate
    return traj


class TestGPSSpoofing:
    """Tests for GPS spoofing injection."""

    def test_shape_preserved(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Output shape should match input shape."""
        modified, labels = injector.inject_gps_spoofing(clean_traj)
        assert modified.shape == clean_traj.shape
        assert labels.shape == (len(clean_traj),)

    def test_labels_correct(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Labels should be 0 before start and 1 after."""
        modified, labels = injector.inject_gps_spoofing(clean_traj, start_idx=100)
        assert labels[:100].sum() == 0
        assert labels[100:].sum() == len(labels) - 100

    def test_position_drifts(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Position should drift progressively from the original."""
        modified, _ = injector.inject_gps_spoofing(clean_traj, start_idx=100, offset_nm=5.0)
        # Early anomalous points should be closer to original than late ones
        early_diff = np.sqrt((modified[110, 0] - clean_traj[110, 0]) ** 2 + (modified[110, 1] - clean_traj[110, 1]) ** 2)
        late_diff = np.sqrt((modified[190, 0] - clean_traj[190, 0]) ** 2 + (modified[190, 1] - clean_traj[190, 1]) ** 2)
        assert late_diff > early_diff

    def test_non_position_unchanged(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Non-position features (alt, vel, hdg, vrate) should be unchanged."""
        modified, _ = injector.inject_gps_spoofing(clean_traj)
        np.testing.assert_array_equal(modified[:, 2:], clean_traj[:, 2:])


class TestPositionJump:
    """Tests for position jump injection."""

    def test_shape_preserved(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Output shape should match input shape."""
        modified, labels = injector.inject_position_jump(clean_traj)
        assert modified.shape == clean_traj.shape

    def test_labels_correct(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Labels should be 0 before jump and 1 from jump onward."""
        modified, labels = injector.inject_position_jump(clean_traj, idx=100)
        assert labels[:100].sum() == 0
        assert labels[100:].all()

    def test_discontinuity(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """There should be a significant position jump at the injection point."""
        modified, _ = injector.inject_position_jump(clean_traj, idx=100, jump_nm=10.0)
        pre_jump = np.sqrt(
            (modified[99, 0] - clean_traj[99, 0]) ** 2 + (modified[99, 1] - clean_traj[99, 1]) ** 2
        )
        post_jump = np.sqrt(
            (modified[100, 0] - clean_traj[100, 0]) ** 2 + (modified[100, 1] - clean_traj[100, 1]) ** 2
        )
        assert pre_jump < 0.01
        assert post_jump > 0.01


class TestGhostAircraft:
    """Tests for ghost aircraft injection."""

    def test_shape_preserved(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Ghost trajectory should have same shape as reference."""
        ghost, labels = injector.inject_ghost_aircraft(clean_traj)
        assert ghost.shape == clean_traj.shape

    def test_all_anomalous(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """All timesteps of a ghost aircraft should be labeled anomalous."""
        _, labels = injector.inject_ghost_aircraft(clean_traj)
        assert labels.sum() == len(labels)

    def test_different_from_original(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Ghost trajectory should differ from the reference."""
        ghost, _ = injector.inject_ghost_aircraft(clean_traj)
        assert not np.allclose(ghost, clean_traj, atol=1.0)


class TestReplayAttack:
    """Tests for replay attack injection."""

    def test_shape_preserved(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Output shape should match input shape."""
        replay_traj = clean_traj.copy()
        replay_traj[:, 0] += 1.0  # Different trajectory
        modified, labels = injector.inject_replay_attack(clean_traj, replay_traj)
        assert modified.shape == clean_traj.shape

    def test_labels_correct(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Labels should mark the replayed section."""
        replay_traj = clean_traj.copy()
        replay_traj[:, 0] += 1.0
        modified, labels = injector.inject_replay_attack(clean_traj, replay_traj, start_idx=100)
        assert labels[:100].sum() == 0
        assert labels[100:].sum() > 0

    def test_prefix_unchanged(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Trajectory before the replay start should be unchanged."""
        replay_traj = clean_traj.copy()
        replay_traj[:, 0] += 1.0
        modified, _ = injector.inject_replay_attack(clean_traj, replay_traj, start_idx=100)
        np.testing.assert_array_equal(modified[:100], clean_traj[:100])


class TestAltitudeManipulation:
    """Tests for altitude manipulation injection."""

    def test_shape_preserved(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Output shape should match input shape."""
        modified, labels = injector.inject_altitude_manipulation(clean_traj)
        assert modified.shape == clean_traj.shape

    def test_labels_correct(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Labels should mark timesteps from start_idx onward."""
        modified, labels = injector.inject_altitude_manipulation(clean_traj, start_idx=100)
        assert labels[:100].sum() == 0
        assert labels[100:].all()

    def test_altitude_changes(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Altitude should progressively deviate from original."""
        modified, _ = injector.inject_altitude_manipulation(
            clean_traj, start_idx=100, alt_offset_ft=2000.0
        )
        early_diff = abs(modified[110, 2] - clean_traj[110, 2])
        late_diff = abs(modified[190, 2] - clean_traj[190, 2])
        assert late_diff > early_diff

    def test_non_altitude_unchanged(self, injector: AnomalyInjector, clean_traj: np.ndarray) -> None:
        """Non-altitude features should be unchanged."""
        modified, _ = injector.inject_altitude_manipulation(clean_traj)
        np.testing.assert_array_equal(modified[:, :2], clean_traj[:, :2])  # lat, lon
        np.testing.assert_array_equal(modified[:, 3:], clean_traj[:, 3:])  # vel, hdg, vrate


class TestGenerateEvaluationSet:
    """Tests for evaluation set generation."""

    def test_generates_mixed_set(self, synthetic_trajectory: np.ndarray) -> None:
        """Should produce a mix of clean and anomalous trajectories."""
        clean_list = [synthetic_trajectory] * 10
        trajs, labels, types = generate_evaluation_set(clean_list, anomalies_per_type=3)
        assert len(trajs) == len(labels) == len(types)
        assert "clean" in types
        assert any(t != "clean" for t in types)

    def test_all_types_present(self, synthetic_trajectory: np.ndarray) -> None:
        """All 5 anomaly types should be present."""
        clean_list = [synthetic_trajectory] * 20
        _, _, types = generate_evaluation_set(clean_list, anomalies_per_type=5)
        unique_types = set(types)
        assert "clean" in unique_types
        assert "gps_spoofing" in unique_types
        assert "position_jump" in unique_types
        assert "ghost_aircraft" in unique_types
        assert "replay_attack" in unique_types
        assert "altitude_manipulation" in unique_types

    def test_label_shapes_match(self, synthetic_trajectory: np.ndarray) -> None:
        """Each label array should match its trajectory length."""
        clean_list = [synthetic_trajectory] * 5
        trajs, labels, _ = generate_evaluation_set(clean_list, anomalies_per_type=2)
        for traj, label in zip(trajs, labels):
            assert len(traj) == len(label)
