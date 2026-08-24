"""
Synthetic policy/FAQ corpus for RAG -- per the blueprint's Section 10
(no client/real policy data, all invented). Small on purpose: enough
variety to prove retrieval actually discriminates between topics, not a
realistic-scale document set.
"""

POLICY_DOCS = [
    {
        "doc_id": "COV-WATER-01",
        "text": (
            "Water damage from a sudden and accidental discharge (e.g. a "
            "burst pipe) is covered under standard homeowners policies. "
            "Damage from gradual leaks, flooding, or lack of maintenance "
            "is NOT covered and requires separate flood insurance."
        ),
    },
    {
        "doc_id": "COV-AUTO-DED-01",
        "text": (
            "Auto collision coverage includes a standard deductible of "
            "$500, applied per incident. Comprehensive coverage (theft, "
            "fire, vandalism) carries a separate $250 deductible."
        ),
    },
    {
        "doc_id": "CLAIMS-PROCESS-01",
        "text": (
            "To file a claim: contact support within 30 days of the "
            "incident, provide a police report if applicable, and submit "
            "photos of the damage. Claims are typically reviewed within "
            "5-7 business days."
        ),
    },
    {
        "doc_id": "CANCEL-POLICY-01",
        "text": (
            "Policies may be cancelled at any time with 30 days written "
            "notice. A pro-rated refund is issued for unused premium; a "
            "$50 early cancellation fee applies within the first 6 months."
        ),
    },
    {
        "doc_id": "COV-THEFT-01",
        "text": (
            "Theft of personal property is covered up to the policy's "
            "stated limit (typically $50,000 for homeowners). High-value "
            "items (jewelry, electronics over $2,000) require a separate "
            "rider to be fully covered."
        ),
    },
    {
        "doc_id": "KYC-VERIFY-01",
        "text": (
            "Identity verification for new policies requires a "
            "government-issued photo ID and proof of address dated within "
            "the last 90 days. Verification is typically completed within "
            "1 business day."
        ),
    },
    {
        "doc_id": "COV-FIRE-01",
        "text": (
            "Fire damage is covered under both homeowners and auto "
            "policies, including damage from wildfires. Smoke damage to "
            "adjacent, undamaged property is also covered under the same "
            "claim."
        ),
    },
    {
        "doc_id": "PREMIUM-PAYMENT-01",
        "text": (
            "Premiums can be paid monthly, quarterly, or annually. Annual "
            "payment receives a 5% discount. A payment more than 15 days "
            "late may result in a lapse in coverage."
        ),
    },
]
