"""Contract tests for the durable append-only rollout store."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
import tempfile
import unittest
import uuid
from dataclasses import FrozenInstanceError
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mca.domain import DomainError, Event
from mca.store import (
    RolloutCorruptionError,
    RolloutStore,
    SessionLockedError,
)


class EventContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = str(uuid.uuid4())

    def test_event_round_trips_the_versioned_json_contract(self) -> None:
        event = Event.create(
            seq=1,
            session_id=self.session_id,
            event_type="session_created",
            payload={"cwd": "/work", "nested": [1, {"ok": True}]},
        )

        encoded = event.to_dict()
        decoded = Event.from_dict(json.loads(json.dumps(encoded)))

        self.assertEqual(decoded, event)
        self.assertEqual(
            set(encoded),
            {
                "version",
                "seq",
                "event_id",
                "timestamp",
                "session_id",
                "type",
                "payload",
            },
        )
        self.assertEqual(encoded["version"], 1)
        self.assertTrue(encoded["timestamp"].endswith("Z"))

    def test_event_is_immutable_and_does_not_expose_mutable_payloads(self) -> None:
        source = {"nested": ["original"]}
        event = Event.create(
            seq=1,
            session_id=self.session_id,
            event_type="example",
            payload=source,
        )
        source["nested"].append("mutated")

        with self.assertRaises(FrozenInstanceError):
            event.seq = 2  # type: ignore[misc]
        with self.assertRaises(TypeError):
            event.payload["other"] = True  # type: ignore[index]
        self.assertEqual(event.to_dict()["payload"], {"nested": ["original"]})

    def test_event_rejects_invalid_common_fields_and_non_json_payloads(self) -> None:
        valid = Event.create(
            seq=1,
            session_id=self.session_id,
            event_type="example",
            payload={},
        ).to_dict()
        invalid_values = {
            "version": 2,
            "seq": 0,
            "event_id": "not-a-uuid",
            "timestamp": "2026-08-27T10:00:00+08:00",
            "session_id": "../escape",
            "type": "",
        }

        for field, value in invalid_values.items():
            with self.subTest(field=field):
                document = dict(valid)
                document[field] = value
                with self.assertRaises(DomainError):
                    Event.from_dict(document)

        for payload in ({"value": object()}, {"value": math.nan}):
            with self.subTest(payload=payload):
                document = dict(valid)
                document["payload"] = payload
                with self.assertRaises(DomainError):
                    Event.from_dict(document)


class RolloutStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.sessions_root = Path(self.temporary_directory.name) / "sessions"
        self.session_id = str(uuid.uuid4())

    @property
    def rollout_path(self) -> Path:
        return self.sessions_root / f"{self.session_id}.jsonl"

    def _event_line(
        self, seq: int, *, session_id: str | None = None, version: int = 1
    ) -> bytes:
        document = Event.create(
            seq=seq,
            session_id=session_id or self.session_id,
            event_type="example",
            payload={"seq": seq},
        ).to_dict()
        document["version"] = version
        return (
            json.dumps(document, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")

    def _write_raw(self, data: bytes) -> None:
        self.sessions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.rollout_path.write_bytes(data)
        os.chmod(self.rollout_path, 0o600)

    def test_append_assigns_strictly_sequential_sequences(self) -> None:
        with RolloutStore.create(self.sessions_root, self.session_id) as store:
            first = store.append("session_created", {"cwd": "/work"})
            second = store.append("turn_started", {"turn_id": "turn-1"})

            self.assertEqual((first.seq, second.seq), (1, 2))
            self.assertEqual([event.seq for event in store.load()], [1, 2])

    def test_jsonl_round_trip_preserves_unicode_and_nested_payloads(self) -> None:
        payload = {
            "message": "你好, agent",
            "values": [1, None, False, {"path": "src/mca.py"}],
        }
        with RolloutStore.create(self.sessions_root, self.session_id) as store:
            expected = store.append("example", payload)

        with RolloutStore.open(self.sessions_root, self.session_id) as reopened:
            self.assertEqual(reopened.load(), [expected])

        raw_lines = self.rollout_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(raw_lines), 1)
        self.assertEqual(json.loads(raw_lines[0]), expected.to_dict())

    def test_new_sessions_directory_and_rollout_file_have_private_modes(self) -> None:
        with RolloutStore.create(self.sessions_root, self.session_id):
            pass

        directory_mode = stat.S_IMODE(self.sessions_root.stat().st_mode)
        file_mode = stat.S_IMODE(self.rollout_path.stat().st_mode)
        self.assertEqual(directory_mode, 0o700)
        self.assertEqual(file_mode, 0o600)

    def test_load_ignores_one_malformed_final_record_and_can_append_after_it(self) -> None:
        with RolloutStore.create(self.sessions_root, self.session_id) as store:
            store.append("example", {"number": 1})
            store.append("example", {"number": 2})

        with self.rollout_path.open("ab") as stream:
            stream.write(b'{"version":1,"seq":3')

        with RolloutStore.open(self.sessions_root, self.session_id) as store:
            self.assertEqual([event.seq for event in store.load()], [1, 2])
            appended = store.append("example", {"number": 3})
            self.assertEqual(appended.seq, 3)

        with RolloutStore.open(self.sessions_root, self.session_id) as store:
            self.assertEqual([event.seq for event in store.load()], [1, 2, 3])

    def test_load_accepts_a_valid_final_record_without_a_newline(self) -> None:
        self._write_raw(self._event_line(1).rstrip(b"\n"))

        with RolloutStore.open(self.sessions_root, self.session_id) as store:
            self.assertEqual([event.seq for event in store.load()], [1])
            self.assertEqual(store.append("example", {}).seq, 2)

        with RolloutStore.open(self.sessions_root, self.session_id) as store:
            self.assertEqual([event.seq for event in store.load()], [1, 2])

    def test_load_rejects_json_corruption_before_the_final_record(self) -> None:
        self._write_raw(
            self._event_line(1) + b"not-json\n" + self._event_line(2)
        )

        with self.assertRaisesRegex(RolloutCorruptionError, "line 2"):
            RolloutStore.open(self.sessions_root, self.session_id)

    def test_load_rejects_middle_semantic_corruption(self) -> None:
        other_session = str(uuid.uuid4())
        corrupt_lines = {
            "version": self._event_line(2, version=99),
            "session": self._event_line(2, session_id=other_session),
        }

        for name, corrupt_line in corrupt_lines.items():
            with self.subTest(name=name):
                self._write_raw(
                    self._event_line(1) + corrupt_line + self._event_line(3)
                )
                with self.assertRaisesRegex(RolloutCorruptionError, "line 2"):
                    RolloutStore.open(self.sessions_root, self.session_id)

    def test_load_rejects_sequence_gaps_and_duplicates(self) -> None:
        invalid_sequences = ((1, 3), (1, 1))

        for sequences in invalid_sequences:
            with self.subTest(sequences=sequences):
                self._write_raw(b"".join(self._event_line(seq) for seq in sequences))
                with self.assertRaisesRegex(RolloutCorruptionError, "sequence"):
                    RolloutStore.open(self.sessions_root, self.session_id)

    def test_load_rejects_semantically_invalid_final_record(self) -> None:
        self._write_raw(self._event_line(1) + self._event_line(2, version=99))

        with self.assertRaisesRegex(RolloutCorruptionError, "version"):
            RolloutStore.open(self.sessions_root, self.session_id)

    def test_session_id_must_be_a_canonical_uuid_before_path_construction(self) -> None:
        invalid_ids = (
            "../escape",
            "not-a-uuid",
            self.session_id.upper(),
            f"{self.session_id}/child",
        )

        for invalid_id in invalid_ids:
            with self.subTest(session_id=invalid_id):
                with self.assertRaises(ValueError):
                    RolloutStore.create(self.sessions_root, invalid_id)

        self.assertFalse(Path(self.temporary_directory.name, "escape.jsonl").exists())

    def test_only_one_writer_can_hold_a_session_lock(self) -> None:
        first = RolloutStore.create(self.sessions_root, self.session_id)
        self.addCleanup(first.close)

        with self.assertRaises(SessionLockedError):
            RolloutStore.open(self.sessions_root, self.session_id)

        first.close()
        with RolloutStore.open(self.sessions_root, self.session_id):
            pass

    def test_append_rejects_an_event_for_another_session_or_sequence(self) -> None:
        with RolloutStore.create(self.sessions_root, self.session_id) as store:
            wrong_session = Event.create(
                seq=1,
                session_id=str(uuid.uuid4()),
                event_type="example",
                payload={},
            )
            wrong_sequence = Event.create(
                seq=2,
                session_id=self.session_id,
                event_type="example",
                payload={},
            )

            with self.assertRaises(ValueError):
                store.append(wrong_session)
            with self.assertRaises(ValueError):
                store.append(wrong_sequence)


if __name__ == "__main__":
    unittest.main()
