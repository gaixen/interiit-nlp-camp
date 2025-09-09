from utils.auth import get_service
import email, os, base64, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class gmailAgent:
    """this class handles Gmail operations.
    this includes operations like reading emails, downloading attachments etc."""
    def __init__(self, download_dir="downloads"):
        self.service = get_service("gmail", "v1")
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            logger.debug("Creating download directory")
            os.makedirs(self.download_dir)


    def list_unread_emails(self, max_results=5):
        """this function lists unread emails in the inbox"""
        results = self.service.users().messages().list(
            userId="me", labelIds=["INBOX"], q="is:unread", maxResults=max_results
        ).execute()
        messages = results.get("messages", [])
        emails = []
        for msg in messages:
            msg_data = self.service.users().messages().get(userId="me", id=msg["id"]).execute()
            payload = msg_data["payload"]
            headers = payload["headers"]
            subject = ""
            sender = ""
            for h in headers:
                if h["name"] == "Subject":
                    subject = h["value"]
                if h["name"] == "From":
                    sender = h["value"]

            snippet = msg_data.get("snippet", "")
            emails.append({"id": msg["id"], "subject": subject, "from": sender, "snippet": snippet})
            logger.info(f"Found unread email: {msg['id']}")

        return emails

    def download_attachments(self, max_results=5):
        """this function downloads attachments from unread emails in the inbox"""
        results = self.service.users().messages().list(
            userId="me", labelIds=["INBOX"], q="has:attachment is:unread", maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        for msg in messages:
            msg_data = self.service.users().messages().get(userId="me", id=msg["id"]).execute()
            parts = msg_data["payload"].get("parts", [])

            for part in parts:
                if part.get("filename"):
                    att_id = part["body"]["attachmentId"]
                    att = self.service.users().messages().attachments().get(
                        userId="me", messageId=msg["id"], id=att_id
                    ).execute()
                    data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))

                    filepath = os.path.join(self.download_dir, part["filename"])
                    with open(filepath, "wb") as f:
                        f.write(data)
                    logger.info(f"Downloaded attachment: {part['filename']}")


# if __name__ == "__main__":
#     agent = gmailAgent()
#     unread_emails = agent.list_unread_emails()
#     for mail in unread_emails:
#         print(f"From: {mail['from']}, Subject: {mail['subject']}, Snippet: {mail['snippet']}")

#     agent.download_attachments()
