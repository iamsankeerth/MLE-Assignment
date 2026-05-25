# MLE Hiring Agent Context

Definitions for the MLE Support Ticket Agent domain.

## Language

**Support Ticket**:
An incoming customer request consisting of a JSON-formatted conversation log (`Issue`), `Subject`, and `Company`. The agent parses this to classify metadata (e.g., `Product Area`, `Request Type`, `Risk Level`) and generates a `Response`.
_Avoid_: Ticket, Email, Message

**Product Area**:
The specific ecosystem a Support Ticket relates to (typically Visa, DevPlatform, or Claude).
_Avoid_: Project, Domain

**Request Type**:
The categorization of the issue. Must be one of: `product_issue`, `feature_request`, `bug`, or `invalid`.
_Avoid_: Issue Type, Category

**Risk Level**:
The potential severity of the ticket. Must be one of: `low`, `medium`, `high`, or `critical`.
_Avoid_: Priority, Severity
