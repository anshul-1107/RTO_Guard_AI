---
title: Order Hold and Cancellation Policy
doc_type: brand_sop
applies_to: all
---
# Order Hold and Cancellation Policy

## Holding an order
An order may be placed on hold for a maximum of 24 hours, for example while waiting for verification, address correction, or payment. Held orders must not be handed to the courier.

## After the hold expires
When a hold expires without customer response, MEDIUM-risk orders are shipped as normal. HIGH-risk orders are escalated to the Ops Lead, who decides whether to ship or cancel.

## Cancellation rules
RTO Guard may never cancel an order on its own. Every cancellation not requested by the customer requires human approval from the Ops Lead. Prepaid orders are never cancelled for RTO risk.

## Reason codes
Use CUST_CANCEL_VERIFY when the customer cancels during verification, OPS_CANCEL_RISK when the Ops Lead cancels a high-risk order, and ADDR_UNSERVICEABLE when the pincode cannot be served.
