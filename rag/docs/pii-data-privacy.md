---
title: PII and Data Privacy Policy
doc_type: brand_sop
applies_to: all
---
# PII and Data Privacy Policy

## Personal data
Customer phone numbers, full addresses, and names are personal data. They must not be sent to external AI services in raw form.

## Masking
Before any order data is sent to an LLM, phone numbers must be masked (for example 98XXXXXX12) and the address must be reduced to pincode, city tier, and address completeness flags. The customer's first name may be used only when drafting the final message.

## Logging
Logs and traces must store masked phone numbers only. Full addresses must never appear in logs.

## Retention
Verification conversation data is retained for 90 days and then deleted.
