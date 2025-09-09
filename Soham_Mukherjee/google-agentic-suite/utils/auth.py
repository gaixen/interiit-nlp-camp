# drive authentication
import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly","https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar","https://www.googleapis.com/auth/drive.file",
]

TOKEN_PATH = os.path.join("config", "token.json")
CREDENTIALS_PATH = os.path.join("config", "auth.json")


def get_service(api_name: str, api_version: str, scopes: list = None) -> object:
    
    if scopes is None:
        scopes = SCOPES #fallback to default scopes

    creds = None

    # Load saved tokens
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, scopes)

    # Refresh or request new tokens
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, scopes)
            creds = flow.run_local_server(port=0)

        # Save tokens
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    # Build service object
    service = build(api_name, api_version, credentials=creds)
    return service


if __name__ == "__main__":
    # Quick test: List next 5 events from Calendar
    service = get_service("calendar", "v3")
    events_result = service.events().list(
        calendarId="primary", maxResults=5, singleEvents=True, orderBy="startTime"
    ).execute()
    events = events_result.get("items", [])

    if not events:
        print("empty..")
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        print(start, event["summary"])
