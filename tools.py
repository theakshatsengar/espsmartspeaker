import datetime
import smtplib
from email.message import EmailMessage
from contacts import CONTACTS
import os
from dotenv import load_dotenv

# Tool registry
TOOLS = {}

load_dotenv()

SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

def tool(func):
    """Decorator to register a function as a tool."""
    TOOLS[func.__name__] = func
    return func

@tool
def get_datetime():
    """Returns the current date and time."""
    return datetime.datetime.now().isoformat()

@tool
def send_email(to: str, subject: str, body: str):
    """Send an email using SMTP with SSL."""
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 465  # Changed from 587 to 465 for SSL

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    msg["Reply-To"] = SMTP_USER

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    return f"Email sent to {to}"

@tool
def send_email_to_contact(contact: str, subject: str, body: str):
    """Send an email to a saved contact."""
    to = CONTACTS.get(contact.lower())
    if not to:
        return f"Contact '{contact}' not found."
    return send_email(to=to, subject=subject, body=body)

# Example: add more tools below using @tool
# @tool
# def web_search(query: str):
#     ...

def execute_tool(tool_call: str):
    """
    Parse and execute a tool call string, e.g., 'get_datetime()' or 'send_email(to="a@b.com", subject="Hi", body="Hello")'
    """
    import ast
    import inspect

    # Parse the function name and arguments
    try:
        node = ast.parse(tool_call, mode='eval').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in TOOLS:
                return f"Tool '{func_name}' not found."
            func = TOOLS[func_name]
            # Build args and kwargs
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)
        else:
            return "Invalid tool call syntax."
    except Exception as e:
        return f"Error executing tool: {str(e)}"
