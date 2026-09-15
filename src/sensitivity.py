"""Data-type sensitivity tiers and cost weights omega(r) for CI-MoE.

Each *generic* data type (data/data_types.csv) is assigned a tier
{high, medium, low}; the tier -> cost weight omega up-weights FALSE GRANTS of
sensitive data in the cost-sensitive objective and defines the high-sensitivity
subset for Sens-FPR. Tiering is a deployment POLICY (see Discussion), not a
universal constant. Anything unlisted defaults to 'medium'.
Coverage over the 76-type catalog: 31 high / 30 medium / 15 low.
"""

OMEGA = {"high": 3.0, "medium": 1.5, "low": 1.0}

HIGH = {
    "Financial Documents", "Income Details", "Payment Details",
    "Payment Card Information", "Payment Method Details", "Bank Account Details",
    "Tax Information", "Transaction History", "Investment Information",
    "Credit Information", "Debt Information", "Expense Data",
    "Account Credentials", "Social Security Number (SSN)",
    "Personal Identification Numbers", "Passport Documents", "Visa Documents",
    "Visa Application History", "Driver License Number", "Date of Birth",
    "Medical History", "Family Medical History", "Test/Diagnostic Results",
    "Medications", "Vital Signs", "Physical Health Conditions",
    "Physical Limitations or Injuries", "Allergies", "Body Weight",
    "Sleep Data", "Child Information",
}
LOW = {
    "Dietary Preferences", "Travel Preferences", "Travel Destination",
    "Travel Activities", "Movie Preferences", "Movie Watching History",
    "Music Listening History", "Hobbies and Interests",
    "Relationship Preferences", "Nutritional Goals", "Fitness Goal",
    "Shopping List", "Pet Ownership", "Slack Workspace Name",
    "Exercise Activity Details",
}


# Keyword fallback for data types NOT in the Wu catalog (e.g. the SPA / Education
# datasets). Exact-match HIGH/LOW above is checked FIRST, so Wu behavior is
# unchanged; keywords only classify otherwise-'medium' unseen types. Substrings,
# lowercased.
HIGH_KW = (
    "financ", "bank", "payment", "card number", "credit", "debt", "tax",
    "income", "salary", "invest", "transaction", "account", "password",
    "credential", "ssn", "social security", "passport", "visa", "licen",
    "date of birth", "medical", "health", "diagnos", "medication",
    "prescription", "vital", "allerg", "sleep", "disease", "biometric",
    "fingerprint", "location", "address", "gps", "geoloc", "door", "lock",
    "camera", "surveillance", "security", "child", "contact", "phone number",
    "email", "message", "call log", "calendar", "photo", "recording", "voice",
)
LOW_KW = (
    "weather", "music", "movie", "film", "song", "news", "hobby", "interest",
    "shopping list", "travel", "vacation", "fitness", "diet", "nutrition",
    "recipe", "joke", "game", "trivia", "sport", "score", "entertainment",
    "audiobook", "podcast", "temperature", "light", "timer", "alarm",
    "reminder", "preference",
)


def tier_of(generic_data_type: str) -> str:
    g = (generic_data_type or "").strip()
    if g in HIGH:
        return "high"
    if g in LOW:
        return "low"
    gl = g.lower()
    if any(k in gl for k in HIGH_KW):
        return "high"
    if any(k in gl for k in LOW_KW):
        return "low"
    return "medium"


def omega_of(generic_data_type: str) -> float:
    return OMEGA[tier_of(generic_data_type)]
