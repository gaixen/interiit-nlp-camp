### requirements: google cloud credentials(json file); google-api-python-client
### uv add google-api-python-client

import os, logging
from googleapiclient.http import MediaFileUpload
from utils.auth import get_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class driveAgent:
    """this class handles Google Drive operations.
    this includes operations like upload, organize, move, rename, delete files and folders."""
    def __init__(self):
        self.service = get_service("drive", "v3")

    def upload_file(self, filepath, folder_id=None):
        """upload a file to Google Drive."""
        file_metadata = {"name": os.path.basename(filepath)}
        if folder_id:
            file_metadata["parents"] = [folder_id]
        media = MediaFileUpload(filepath, resumable=True)
        uploaded_file = self.service.files().create(body=file_metadata,
            media_body=media,fields="id, name, parents, webViewLink"
        ).execute()
        logger.info(f"{uploaded_file['name']}':{uploaded_file['webViewLink']}")
        return uploaded_file

    def get_or_create_folder(self, folder_name, parent_id=None):
        """create a folder in Google Drive or find it if it exists."""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id is not None:
            query += f" and '{parent_id}' in parents"
            logger.info(f"query: {query}")

        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get("files", [])
        if folders:
            logger.info(f"Found existing folder: {folder_name} ({folders[0]['id']})")
            return folders[0]["id"]
        # create folder otherwise
        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder"
        }
        if parent_id is not None:
            folder_metadata["parents"] = [parent_id]
            logger.info(f"parent_id: {parent_id}")
        folder = self.service.files().create(body=folder_metadata, fields="id").execute()
        logger.info(f"new folder: {folder_name} ({folder['id']})")
        return folder["id"]


    def organize_file(self, file_id, category="Misc"):
        """move a file to a specific category folder."""
        folder_id = self.get_or_create_folder(category)

        # Move file into folder
        file = self.service.files().get(fileId=file_id, fields="parents").execute()
        previous_parents = ",".join(file.get("parents", []))

        updated_file = self.service.files().update(fileId=file_id,
            addParents=folder_id,removeParents=previous_parents,fields="id, parents"
        ).execute()

        logger.info(f"file location changes {file_id} :folder {category}")
        return updated_file


    def move_file(self, file_id, new_folder_id):
        """change location of a file in Drive."""

        file = self.service.files().get(fileId=file_id, fields="parents").execute()
        previous_parents = ",".join(file.get("parents", []))

        updated_file = self.service.files().update(
            fileId=file_id,
            addParents=new_folder_id,
            removeParents=previous_parents,
            fields="id, parents"
        ).execute()
        return updated_file


    def rename_file(self, file_id, new_name):
        """rename a file in Drive."""
        updated_file = self.service.files().update(
            fileId=file_id,
            body={"name": new_name},
            fields="id, name"
        ).execute()

        logger.info(f"file renamed:{updated_file['name']}")
        return updated_file


    def delete_file(self, file_id):
        """permanently delete a file in Drive."""
        self.service.files().delete(fileId=file_id).execute()
        logger.info(f"file deleted: {file_id}")


# if __name__ == "__main__":
    # Move file
    # move_file(test_file_id, test_folder_id)
    # Rename file
    # rename_file(test_file_id, "Renamed_Document.pdf")
    # Delete file
    # delete_file(test_file_id)
