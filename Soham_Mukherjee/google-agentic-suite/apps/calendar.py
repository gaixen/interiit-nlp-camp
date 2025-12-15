### requuirements: google cloud credentials(json file)

from utils.auth import get_service
from datetime import datetime, timedelta
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class calendarAgent:
    def __init__(self):
        self.service = get_service("calendar", "v3")


    def create_event(self, summary="New Event", description="Created automatically")-> Any:
        """create a new event in the primary calendar."""
        start_time = datetime.utcnow() + timedelta(hours=1)
        end_time = start_time + timedelta(hours=1)

        event = {
            "summary": summary,"description": description,
            "start": {"dateTime": start_time.isoformat() + "Z", "timeZone": "UTC"},
            "end": {"dateTime": end_time.isoformat() + "Z", "timeZone": "UTC"},
        }

        event_result = self.service.events().insert(calendarId="primary", body=event).execute()
        logger.info(f"event creation done: {event_result['htmlLink']}")
        return event_result


    def list_events(self, max_results=5)->Any:
        """list upcoming events from the primary calendar."""
        events_result = self.service.events().list(
            calendarId="primary",
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        if not events:
            logger.debug("No upcoming events found.")
            return []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            logger.info(f"event: {event['summary']} at {start}")
        return events


    def update_event(self, event_id, new_summary=None, new_description=None)->Any:
        """update an existing event in the primary calendar."""
        event = self.service.events().get(calendarId="primary", eventId=event_id).execute()

        if new_summary:
            event["summary"] = new_summary
        if new_description:
            event["description"] = new_description

        updated_event = self.service.events().update(
            calendarId="primary", eventId=event_id, body=event
        ).execute()
        logger.info(f"event update done: {updated_event['htmlLink']}")
        return updated_event


    def delete_event(self, event_id)->None:
        """delete an event in the primary calendar."""

        self.service.events().delete(calendarId="primary", eventId=event_id).execute()
        # no particular return value

# if __name__ == "__main__":
#     agent = calendarAgent()
#     new_event = agent.create_event("Demo Meeting", "Test event from API")
#     agent.list_events()

