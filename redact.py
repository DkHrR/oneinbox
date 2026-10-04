import re

def redact_text(text: str) -> str:
    """
    Strips names, phone numbers, OTPs, URLs, account numbers from text.
    """
    if not text:
        return text
    
    # Redact URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Redact Email Addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[EMAIL]', text)
    
    # Redact phone numbers (international and US formats)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    
    # Redact Account Numbers / Credit Cards (9 to 19 digits)
    text = re.sub(r'\b(?:\d[-.\s]?){9,19}\b', '[ACCOUNT/CARD]', text)
    
    # Redact OTPs (typically 4 to 8 digit numbers in isolation)
    # We do this after account numbers so longer strings are already replaced.
    text = re.sub(r'\b\d{4,8}\b', '[OTP/CODE]', text)
    
    # Redact Names after common greetings
    def replace_greeting(match):
        greeting = match.group(1)
        return f"{greeting} [NAME]"
    
    text = re.sub(r'\b(Dear|Hi|Hello|Hey)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', replace_greeting, text)

    return text

if __name__ == "__main__":
    sample = "Hi John Doe, your OTP is 123456. Call me at 555-123-4567. Account 1234567890. Link: https://example.com"
    print("Original:", sample)
    print("Redacted:", redact_text(sample))
