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

    def test_unreadable_dial_blocks_the_call(self):
        """If the dial cannot be read, the cloud call is refused (D.2).

        This test previously asserted the opposite — that an unreadable config
        let the call through, filed under graceful degradation. That was the
        wrong invariant for this seam. Degradation applies to features: when a
        model-judgment feature fails, June does less and says so. This is a
        safety check, and a safety check that cannot evaluate itself has not
        established that the action is permitted.

        The user-visible cost of the two failure directions is not symmetric. A
        false "blocked" is a turn that degrades and explains itself. A false
        "permitted" is data leaving a machine whose owner set the dial to stop
        exactly that, with a provenance frame that does not mention it.
        """
        event = CloudCallEvent(model_id="gemini-2.0-flash", phase="start", payload_summary="test")
        with patch(
            "june_brain.config_store.get_privacy_dial",
            side_effect=ImportError("no config"),
        ):
            with patch("june_brain.providers.provenance._record_egress_to_ledger"):
                with pytest.raises(CloudEgressBlockedError):
                    record_cloud_call(event)
