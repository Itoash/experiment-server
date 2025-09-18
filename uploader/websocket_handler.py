from fastapi import WebSocket, WebSocketDisconnect
import os
import base64
from typing import Dict, Any
from time import time
from file_utils import IMAGES_DIR
import logging
import shutil
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the RAW_DIR exists
RAW_DIR = os.path.join(IMAGES_DIR, "raws")


# Create the directory if it doesn't exist
os.makedirs(RAW_DIR, exist_ok=True)


class DirectoryUploadManager:
    def __init__(self):
        self.active_uploads: Dict[str, Dict[str, Any]] = {}
    
    async def handle_upload(self, websocket: WebSocket, upload_id: str, upload_name: str, overwrite: bool):
        """Handle the directory upload process"""
        try:
            # Sanitize the upload name for filesystem safety
            safe_upload_name = self._sanitize_directory_name(upload_name)

            # Create directory under RAW_DIR
            upload_dir = os.path.join(RAW_DIR, safe_upload_name)
            os.makedirs(upload_dir, exist_ok=True)

           
            # If overwrite, remove existing directory and recreate
            if overwrite and os.path.exists(upload_dir):
                logger.info(f"Overwriting existing directory: {upload_dir}")
                shutil.rmtree(upload_dir)
                
            
            await websocket.send_json({
                "type": "ready",
                "message": f"Ready to receive files for '{upload_name}'",
                "upload_id": upload_id,
                "upload_dir": str(upload_dir)
            })
            
            self.active_uploads[upload_id] = {
                "upload_dir": upload_dir,
                "upload_name": upload_name,
                "files_received": 0,
                "total_size": 0,
                "status": "active"
            }
            
            while True:
                try:
                    data = await websocket.receive_json()
                    
                    if data["type"] == "file":
                        await self._save_file(data, upload_id, websocket)
                    elif data["type"] == "complete":
                        await self._complete_upload(upload_id, websocket)
                        break
                    elif data["type"] == "ping":
                        await websocket.send_json({"type": "pong"})
                
                except WebSocketDisconnect:
                    print(f"WebSocket disconnected for upload {upload_id}")
                    break
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing upload: {str(e)}"
                    })
                    break
        
        finally:
            if upload_id in self.active_uploads:
                self.active_uploads[upload_id]["status"] = "completed"

    
    def _sanitize_directory_name(self, name: str) -> str:
        """Sanitize directory name to prevent path traversal and invalid characters"""
        import re
        
        # Remove or replace dangerous characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Windows forbidden chars
        safe_name = re.sub(r'\.\.', '_', safe_name)      # Prevent directory traversal
        safe_name = safe_name.strip('. ')                # Remove leading/trailing dots/spaces
        
        # Ensure it's not empty after sanitization
        if not safe_name:
            safe_name = f"upload_{int(time())}"

        return safe_name
    async def _save_file(self, data: dict, upload_id: str, websocket: WebSocket):
        """Save individual file maintaining directory structure"""
        try:
            file_path = data["path"]
            file_content = base64.b64decode(data["content"])

            full_path = os.path.join(self.active_uploads[upload_id]["upload_dir"], file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(file_content)
            
            self.active_uploads[upload_id]["files_received"] += 1
            self.active_uploads[upload_id]["total_size"] += len(file_content)
            
            await websocket.send_json({
                "type": "progress",
                "files_received": self.active_uploads[upload_id]["files_received"],
                "total_size": self.active_uploads[upload_id]["total_size"],
                "current_file": file_path
            })
        
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Error saving file {data.get('path', 'unknown')}: {str(e)}"
            })
    
    async def _complete_upload(self, upload_id: str, websocket: WebSocket):
        """Complete the upload process"""
        upload_info = self.active_uploads[upload_id]
        await websocket.send_json({
            "type": "complete",
            "message": "Upload completed successfully",
            "files_received": upload_info["files_received"],
            "total_size": upload_info["total_size"],
            "upload_dir": str(upload_info["upload_dir"])
        })
