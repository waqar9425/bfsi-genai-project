from argus.guardrails import redact_pii


def test_redacts_ssn():
    text, found = redact_pii("my SSN is 123-45-6789, please help")
    assert "123-45-6789" not in text
    assert "[REDACTED_SSN]" in text
    assert found == ["SSN"]


def test_redacts_email():
    text, found = redact_pii("contact me at jane.doe@example.com")
    assert "jane.doe@example.com" not in text
    assert "EMAIL" in found


def test_redacts_credit_card():
    text, found = redact_pii("card number 4111-1111-1111-1111 was charged")
    assert "4111-1111-1111-1111" not in text
    assert "CREDIT_CARD" in found


def test_redacts_multiple_types_in_one_message():
    text, found = redact_pii("SSN 123-45-6789 and email a@b.com")
    assert set(found) == {"SSN", "EMAIL"}
    assert "123-45-6789" not in text
    assert "a@b.com" not in text


def test_clean_text_is_unchanged():
    text, found = redact_pii("I want to check my claim status")
    assert text == "I want to check my claim status"
    assert found == []
