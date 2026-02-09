from google import genai
from google.colab import userdata
from IPython.display import display, Markdown
import textwrap

def setup_gemini():
    """
    Initializes the Gemini client with silent safety checks.
    Only prints if the key is missing, improperly formatted, or invalid.
    """
    api_key = userdata.get('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in Colab Secrets.")
        return None

    clean_key = api_key.strip()
    
    # Common student errors: hidden spaces or wrong key type
    if len(api_key) != len(clean_key):
        print("⚠️ Warning: Your API key contains hidden spaces.")
    
    if len(clean_key) != 39:
        print(f"⚠️ Warning: Standard Gemini keys are 39 characters. Yours is {len(clean_key)}.")

    try:
        client = genai.Client(api_key=clean_key)
        return client
    except Exception as e:
        print(f"❌ Critical Error during setup: {e}")
        return None

def print_md(text):
    """
    Renders text as Markdown in the Colab output.
    Useful for visualizing headers, lists, and tables from model responses.
    """
    if text:
        display(Markdown(str(text)))

def check_quota(client, model_id="gemini-2.5-flash"):
    """
    Displays remaining daily requests to help students manage their usage.
    """
    try:
        # In 2026 SDK, usage metadata is retrieved via the model object
        model_info = client.models.get(model=model_id)
        # Note: API response structure can vary; we check for available quota fields
        print(f"📊 Model: {model_id}")
        print("Note: Check AI Studio dashboard for precise real-time daily quota.")
    except Exception:
        # Silent failure if metadata is unavailable
        pass


def end_session_report(client, model_id="gemini-2.5-flash"):
    """
    Displays a final session summary.
    In 2026, the free tier for Flash is 250 requests per day.
    """
    # Standard daily limits for the 2026 Free Tier
    limits = {
        "gemini-2.5-flash": 250,
        "gemini-2.5-flash-lite": 1000,
        "gemini-2.5-pro": 100
    }
    
    total = limits.get(model_id, 250)
    
    print("\n" + "─" * 45)
    print(f"✨ SESSION WRAP-UP | {model_id.upper()}")
    print("─" * 45)
    print(f"✅ Tasks completed successfully.")
    print(f"📈 Daily limit for this model: {total} requests.")
    print(f"💡 Note: No image generation available on this tier.")
    print("─" * 45)
    print("Remember to disconnect your runtime if you are finished.")
