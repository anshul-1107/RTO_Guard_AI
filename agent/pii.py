"""
PII handling. The LLM never sees phone numbers, names or full addresses.
It writes "{first_name}" and we fill the real name locally after generation.
"""

from __future__ import annotations

import re

PII_FIELDS = {"customer_phone", "customer_name", "address_text"}


def mask_phone(phone: str) -> str:
    return phone[:2] + "X" * (len(phone) - 4) + phone[-2:] if len(phone) >= 6 else "XXXX"


def address_flags(address: str, has_landmark: bool) -> dict:
    return {
        "has_house_no": "H.No" in address
        or bool(re.search(r"\b(flat|house|#)\s*\d", address, re.I)),
        "has_landmark": bool(has_landmark),
        "address_len": len(address),
    }


def mask_order(order: dict) -> dict:
    masked = {k: v for k, v in order.items() if k not in PII_FIELDS}
    masked["customer_phone_masked"] = mask_phone(str(order.get("customer_phone", "")))
    masked["address"] = address_flags(
        str(order.get("address_text", "")), bool(order.get("address_has_landmark"))
    )
    return masked


def fill_name(text: str, first_name: str) -> str:
    return text.replace("{first_name}", first_name)


def contains_pii(text: str, order: dict) -> bool:
    phone = str(order.get("customer_phone", ""))
    addr = str(order.get("address_text", ""))
    return (len(phone) >= 6 and phone in text) or (len(addr) > 10 and addr in text)
