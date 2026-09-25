---
title: COD Order Verification SOP
doc_type: brand_sop
applies_to: all
---
# COD Order Verification SOP

## When verification is required
Every COD order scored MEDIUM or HIGH risk by RTO Guard must be verified before it is handed to the courier. LOW-risk COD orders ship without verification. Prepaid orders are never verified.

## Verification channels
The first attempt is always a WhatsApp confirmation message with two quick-reply buttons: "Confirm order" and "Cancel order". If the customer does not respond within 4 hours, a second attempt is made through an automated IVR call. HIGH-risk orders above ₹3,000 get a human agent call-back instead of IVR.

## Attempt limits and timing
A maximum of 2 verification attempts are allowed per order. Attempts may only be made between 9:00 AM and 9:00 PM IST. Orders placed outside these hours are queued and the first attempt is sent at 9:00 AM.

## Outcomes
If the customer confirms, the order ships immediately. If the customer cancels, the order is cancelled with reason code CUST_CANCEL_VERIFY. If there is no response after 2 attempts, the order moves to a 24-hour hold as described in the Order Hold Policy.
