"""Tests for cloud egress enforcement gate (provenance module)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from june_brain.providers.provenance import (
    CloudCallEvent,
    CloudEgressBlockedError,
    record_cloud_call,
)


class TestRecordCloudCallEnforcement:
    def test_blocks_start_phase_when_local_only(self):
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="start", payload_summary="test")
        with patch(
            "june_brain.config_store.get_privacy_dial",
        ) as mock_dial:
            from june_brain.routing import UserPrivacyDial

            mock_dial.return_value = UserPrivacyDial.LOCAL_ONLY
            with pytest.raises(CloudEgressBlockedError, match="blocked"):
                record_cloud_call(event)

    def test_allows_start_phase_when_not_local_only(self):
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="start", payload_summary="test")
        with patch("june_brain.config_store.get_privacy_dial") as mock_dial:
            from june_brain.routing import UserPrivacyDial

            mock_dial.return_value = UserPrivacyDial.PRIVATE_BY_DEFAULT
            with patch("june_brain.providers.provenance._record_egress_to_ledger"):
                record_cloud_call(event)  # should not raise

    def test_allows_end_phase_even_when_local_only(self):
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="end", payload_summary="test")
        with patch("june_brain.config_store.get_privacy_dial") as mock_dial:
            from june_brain.routing import UserPrivacyDial

            mock_dial.return_value = UserPrivacyDial.LOCAL_ONLY
            with patch("june_brain.providers.provenance._record_egress_to_ledger"):
                record_cloud_call(event)  # end phase should not be blocked

    def test_error_message_includes_model_id(self):
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="start", payload_summary="test")
        with patch("june_brain.config_store.get_privacy_dial") as mock_dial:
            from june_brain.routing import UserPrivacyDial

            mock_dial.return_value = UserPrivacyDial.LOCAL_ONLY
            with pytest.raises(CloudEgressBlockedError, match="gemini-2.0-flash"):
                record_cloud_call(event)

    def test_error_message_suggests_fix(self):
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="start", payload_summary="test")
        with patch("june_brain.config_store.get_privacy_dial") as mock_dial:
            from june_brain.routing import UserPrivacyDial

            mock_dial.return_value = UserPrivacyDial.LOCAL_ONLY
            with pytest.raises(CloudEgressBlockedError, match="Switch to private_by_default"):
                record_cloud_call(event)

    def test_import_error_falls_back_to_not_local_only(self):
        """If config_store is unavailable, _is_local_only returns False."""
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="start", payload_summary="test")
        with patch(
            "june_brain.config_store.get_privacy_dial",
            side_effect=ImportError("no config"),
        ):
            with patch("june_brain.providers.provenance._record_egress_to_ledger"):
                record_cloud_call(event)  # should not raise — graceful degradation
