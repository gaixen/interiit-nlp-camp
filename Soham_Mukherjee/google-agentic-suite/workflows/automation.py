import re, os
from apps.gmail import gmailAgent
from apps.calendar import calendarAgent
from apps.drive import driveAgent

class workFlows:
    def __init__(self):
        self.gmailAgent = gmailAgent()
        self.calendarAgent = calendarAgent()
        self.driveAgent = driveAgent()
        self.download_dir = "downloads"
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def extract_event_details(self, email_text):
            """extract event details from email text using regex."""
            date_pattern = r"(\d{4}-\d{2}-\d{2})"
            time_pattern = r"(\d{1,2}:\d{2})"
            date_match = re.search(date_pattern, email_text)
            time_match = re.search(time_pattern, email_text)
            summary = "Auto-created Event"
            description = email_text[:200] 

            if date_match and time_match:
                summary = f"Meeting on {date_match.group(1)} at {time_match.group(1)}"

            return summary, description


    def gmail_to_calendar(self, max_results = 3):
        """reads unread emails and creates calendar events based on email content."""
        unread_emails = self.gmailAgent.list_unread_emails(max_results=max_results)

        if not unread_emails:
            print("empty..")
            return

        for email in unread_emails:
            summary, description = self.extract_event_details(email["snippet"])
            print(f"emial processing: {email['subject']}")
            calendarAgent.create_event(summary=summary, description=description)

    def categorize_file(self, filename):
        """file categorization based on extension."""
        ext = filename.split(".")[-1].lower()
        if ext in ["pdf"]:
            return "PDFs"
        elif ext in ["jpg", "jpeg", "png"]:
            return "Images"
        elif ext in ["doc", "docx"]:
            return "Documents"
        else:
            return "Misc"


    def gmail_to_drive(self, max_results=3):
        """download attachments from Gmail and upload to Drive."""
        download_dir = self.download_dir
        # if not os.path.exists(download_dir):
        #     os.makedirs(download_dir)

        self.gmailAgent.download_attachments(max_results=max_results)

        for filename in os.listdir(download_dir):
            filepath = os.path.join(download_dir, filename)
            if os.path.isfile(filepath):
                print(f"⬆️ Uploading {filename} to Drive...")
                uploaded_file = self.driveAgent.upload_file(filepath)
                category = self.categorize_file(filename)
                self.driveAgent.organize_file(uploaded_file["id"], category=category)

