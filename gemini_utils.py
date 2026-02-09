from google import genai
from google.colab import userdata

def setup_gemini():
    """
    Initializes the Gemini client with silent safety checks.
    Only prints if the key is missing, improperly formatted, or invalid.
    """
    api_key = userdata.get('GEMINI_API_KEY')
    
    # 1. Check if key is completely missing
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in Colab Secrets.")
        return None

    clean_key = api_key.strip()
    
    # 2. Check for hidden spaces (common copy-paste issue)
    if len(api_key) != len(clean_key):
        print("⚠️ Warning: Your API key contains hidden spaces.")
    
    # 3. Check for standard 39-character length
    if len(clean_key) != 39:
        print(f"⚠️ Warning: Standard Gemini keys are 39 characters. Yours is {len(clean_key)}.")

    try:
        # Initialize client with the cleaned key
        client = genai.Client(api_key=clean_key)
        return client
    except Exception as e:
        # Only print on actual connection or initialization errors
        print(f"❌ Critical Error during setup: {e}")
        return None


def check_quota(client):
    """
    Prints the remaining daily requests for the current model.
    Note: Requires a valid initialized client.
    """
    try:
        # Fetches the specific usage metadata for the free tier
        quota_info = client.models.get_metadata(model="gemini-2.5-flash")
        
        remaining = quota_info.get('remaining_requests_day', 'Unknown')
        total = quota_info.get('total_requests_day', 'Unknown')
        
        print(f"📊 Quota Status: {remaining} / {total} requests remaining for today.")
    except Exception as e:
        # Silent failure if the metadata endpoint is throttled
        pass
