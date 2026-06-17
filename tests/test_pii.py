from agent_firewall.detectors import pii


def test_masks_email_and_keeps_findings():
    text = "Contact me at alice.smith@example.com please."
    masked, findings = pii.mask(text)
    assert "alice.smith@example.com" not in masked
    assert "[REDACTED_EMAIL]" in masked
    assert any(f.evidence["category"] == "email" for f in findings)


def test_credit_card_luhn_filters_false_positive():
    # Valid Visa test number (passes Luhn).
    valid = "My card is 4111 1111 1111 1111."
    # Same as the valid Visa but last digit changed → fails Luhn.
    invalid = "Order number 4111 1111 1111 1112."
    assert any(f.evidence["category"] == "credit_card" for f in pii.scan(valid))
    assert not any(f.evidence["category"] == "credit_card" for f in pii.scan(invalid))


def test_api_token_is_critical_and_redacted():
    masked, findings = pii.mask("token: sk-abcdef0123456789abcdef")
    assert "sk-abcdef0123456789abcdef" not in masked
    assert any(f.severity.value == "critical" for f in findings)


def test_preview_does_not_leak_full_value():
    _, findings = pii.mask("ssn 123-45-6789")
    match_preview = findings[0].evidence["match"]
    assert "123-45-6789" != match_preview
    assert "*" in match_preview


def test_category_filter():
    text = "email a@b.com phone 010-1234-5678"
    findings = pii.scan(text, categories=["email"])
    assert {f.evidence["category"] for f in findings} == {"email"}
