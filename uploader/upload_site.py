import os
import logging
import json
import tempfile
import shutil
import base64
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Form, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional
from file_utils import  DATA_DIR, META_TEMPLATE,IMAGES_DIR, save_processed_images, get_existing_experiments, save_metadata, save_data, delete_experiment
from urllib.parse import urlencode
from db_operations import load_experiment_complete, update_experiment as update_experiment_db
from websocket_handler import DirectoryUploadManager
from threading import Lock
from typing import Dict, List, Optional
from file_utils import  DATA_DIR, META_TEMPLATE,IMAGES_DIR, save_processed_images, get_existing_experiments, save_metadata, save_data, delete_experiment
from urllib.parse import urlencode
from db_operations import load_experiment_complete, update_experiment as update_experiment_db
from websocket_handler import DirectoryUploadManager
from threading import Lock


experiment_locks = {}


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global upload manager
upload_manager = DirectoryUploadManager()

@app.get("/", response_class=HTMLResponse)
def main(request: Request, message: Optional[str] = None, success: Optional[bool] = None):
    return templates.TemplateResponse(
        "upload_form.html",
        {"request": request, "message": message, "success": success}
    )

def _get_parents(paths:List[Dict[str, str]]) -> List[str]:
    parents = set()
    for p in paths:
        relative_path = p.get("relativePath", "")
        if '/' in relative_path:
            parent = os.path.basename(os.path.dirname(relative_path))
            if parent:
                parents.add(parent)
    return list(parents)

@app.post("/upload/", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    experiment_name: str = Form(...),
    start_time: str = Form(...),
    end_time: Optional[str] = Form(None),
    cell_type: str = Form(...),
    condition: Optional[str] = Form(None),
    condition_amount: Optional[str] = Form(None),  # Changed to str to handle empty values
    condition_unit: Optional[str] = Form(None),
    condition_time: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    experiment_files: List[UploadFile] = File(...),
    ac_images: List[UploadFile] = File(default=[]),
    dc_images: List[UploadFile] = File(default=[]),
    seg_images: List[UploadFile] = File(default=[]),
    raw_image_paths: Optional[str] = Form(None),  # JSON string with webkitRelativePath data
    websocket_upload_id: Optional[str] = Form(None),
    websocket_upload_completed: str = Form("false")
):
    if experiment_name not in experiment_locks:
        experiment_locks[experiment_name] = Lock()
    
    with experiment_locks[experiment_name]:
        from datetime import datetime
        try:

            # Handle WebSocket uploaded files first
            raw_images_final_dir = None
            if websocket_upload_id and websocket_upload_completed == "true":
                # With the new direct-save approach, files are already in final location
                # Just verify the upload session exists and get the directory
                if websocket_upload_id in upload_manager.active_uploads:
                    raw_images_final_dir = upload_manager.active_uploads[websocket_upload_id]["upload_dir"]
                    logging.info(f"WebSocket upload completed, files saved to: {raw_images_final_dir}")
                else:
                    logging.warning(f"WebSocket upload {websocket_upload_id} not found in active uploads")
                    raw_images_final_dir = None
            
            # Get existing experiment metadata
            experiments = get_existing_experiments()
            if experiment_name  in experiments:
                params = urlencode({"message": f"Experiment '{experiment_name}' already exists", "success": "0"})
                return RedirectResponse(url=f"/?{params}", status_code=303)
            
            
            # Parse condition_amount if provided
            parsed_condition_amount = None
            if condition_amount and condition_amount.strip():
                try:
                    parsed_condition_amount = float(condition_amount)
                except ValueError:
                    parsed_condition_amount = None


            

            # Save image files to images directory
            saved_ac = save_processed_images(ac_images, "AC", experiment_name)
            saved_dc = save_processed_images(dc_images, "DC", experiment_name)
            saved_seg = save_processed_images(seg_images, "seg", experiment_name)

            # Save raw images to raw directory with path information
            path_data = []
            if raw_image_paths:
                try:
                    path_data = json.loads(raw_image_paths)
                    logging.info(f"Received {len(path_data)} path entries for raw images")
                    logging.info(f"Raw image paths JSON size: {len(raw_image_paths)} characters")
                    if len(path_data) > 0:
                        logging.info(f"Sample path data: {path_data[:3]}")  # Log first 3 entries
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse raw_image_paths JSON: {str(e)}")
                    logging.warning(f"Raw paths data preview: {raw_image_paths[:500]}...")  # Show first 500 chars
            else:
                logging.warning("No raw_image_paths received from frontend")
            
            saved_raw = _get_parents(path_data)
            if raw_image_paths and len(path_data) == 0:
                logging.warning("raw_image_paths is present but contains no valid entries")
            
            # Save metadata JSON with all form data
            metadata = META_TEMPLATE.copy()
            # Clean up and format datetime fields
            formatted_start_time = None if not start_time or start_time.strip() == '' else f"{start_time}:00Z"
            formatted_end_time = None if not end_time or end_time.strip() == '' else f"{end_time}:00Z"
            formatted_condition_time = None if not condition_time or condition_time.strip() == '' else f"{condition_time}:00Z"
            
            metadata.update({
                "last_operation": "create",
                "experiment_name": experiment_name,
                "start_time": formatted_start_time,
                "end_time": formatted_end_time,
                "cell_type": cell_type,
                "condition": condition,
                "condition_amount": parsed_condition_amount,
                "condition_unit": condition_unit,
                "condition_time": formatted_condition_time,
                "notes": notes,
                "images": {
                    "ac": saved_ac,
                    "dc": saved_dc,
                    "segmentation": saved_seg,
                    "raw": saved_raw
                },
                "original_timestamp": datetime.now().isoformat() + "Z",
                "last_updated": datetime.now().isoformat() + "Z"
            })
            
            # Save experiment data files to uploads directory
            saved_data_files = save_data(experiment_files[0], experiment_name)
            saved_metadata = save_metadata(metadata)

            # Load experiment into database
            load_experiment_complete(os.path.join(DATA_DIR, experiment_name))
            logging.info(f"Experiment '{experiment_name}' uploaded successfully with {len(saved_data_files)} data files and {len(saved_ac) + len(saved_dc) + len(saved_seg) + len(saved_raw)} images.")

            # Create summary message
            total_images = len(saved_ac) + len(saved_dc) + len(saved_seg) + len(saved_raw)
            msg = f"Experiment '{experiment_name}' uploaded: {len(saved_data_files)} data files, {total_images} images."
            params = urlencode({"message": msg, "success": "1"})
            return RedirectResponse(url=f"/?{params}", status_code=303)

        except Exception as e:
            logging.error(f"Upload failed: {e}")
            delete_experiment(experiment_name)
            params = urlencode({"message": f"Upload failed: {str(e)}", "success": "0"})
            return RedirectResponse(url=f"/?{params}", status_code=303)


    
    return experiments




@app.websocket("/ws/upload/{upload_id}")
async def websocket_upload(websocket: WebSocket, upload_id: str, upload_name: str = None, overwrite: bool = False):
    await websocket.accept()
    
    if not upload_name:
        await websocket.send_json({"type": "error", "message": "Upload name is required"})
        return

    try:
        # Use the DirectoryUploadManager's handle_upload method which saves directly to final location
        await upload_manager.handle_upload(websocket, upload_id, upload_name, overwrite)
        
    except Exception as e:
        logging.error(f"WebSocket upload error for {upload_id}: {e}")
        # Cleanup on error
        upload_manager.cleanup_upload(upload_id)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            # WebSocket might be closed
            pass
@app.get("/edit", response_class=HTMLResponse)
def edit_experiments_page(request: Request, message: Optional[str] = None, success: Optional[bool] = None):
    """Display page to select experiment for editing"""
    experiments = get_existing_experiments()
    return templates.TemplateResponse(
        "edit_experiments.html",
        {"request": request, "experiments": experiments, "message": message, "success": success}
    )

@app.get("/edit/{experiment_name}", response_class=HTMLResponse)
def edit_experiment_form(request: Request, experiment_name: str, message: Optional[str] = None, success: Optional[bool] = None):
    """Display form to edit specific experiment"""
    experiments = get_existing_experiments()
    if experiment_name not in experiments:
        params = urlencode({"message": f"Experiment '{experiment_name}' not found", "success": "0"})
        return RedirectResponse(url=f"/edit?{params}", status_code=303)
    
    return templates.TemplateResponse(
        "edit_form.html",
        {
            "request": request, 
            "experiment": experiments[experiment_name],
            "experiment_name": experiment_name,
            "message": message, 
            "success": success
        }
    )


    
@app.post("/edit/{experiment_name}", response_class=HTMLResponse)
async def update_experiment(
    request: Request,
    experiment_name: str,
    start_time: str = Form(...),
    end_time: Optional[str] = Form(None),
    cell_type: str = Form(...),
    condition: Optional[str] = Form(None),
    condition_amount: Optional[str] = Form(None),
    condition_unit: Optional[str] = Form(None),
    condition_time: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    experiment_files: List[UploadFile] = File(default=[]),
    experiment_files_mode: Optional[str] = Form(None),  # Add mode toggle
    ac_images: List[UploadFile] = File(default=[]),
    ac_images_mode: Optional[str] = Form(None),  # Add mode toggle
    dc_images: List[UploadFile] = File(default=[]),
    dc_images_mode: Optional[str] = Form(None),  # Add mode toggle
    seg_images: List[UploadFile] = File(default=[]),
    seg_images_mode: Optional[str] = Form(None),  # Add mode toggle
    raw_image_paths: Optional[str] = Form(None),  # JSON string with webkitRelativePath data
    raw_images_mode: Optional[str] = Form(None)  # Add mode toggle
):
    if experiment_name not in experiment_locks:
        experiment_locks[experiment_name] = Lock()
    
    with experiment_locks[experiment_name]:
        from datetime import datetime
        try:
            # Get existing experiment metadata
            experiments = get_existing_experiments()
            if experiment_name not in experiments:
                params = urlencode({"message": f"Experiment '{experiment_name}' not found", "success": "0"})
                return RedirectResponse(url=f"/edit?{params}", status_code=303)

            existing_metadata = experiments[experiment_name]["metadata"] or {}

            # Parse condition_amount if provided
            parsed_condition_amount = None
            if condition_amount and condition_amount.strip():
                try:
                    parsed_condition_amount = float(condition_amount)
                except ValueError:
                    parsed_condition_amount = None
            
            # Save raw images to raw directory with path information
            path_data = []
            if raw_image_paths:
                try:
                    path_data = json.loads(raw_image_paths)
                    logging.info(f"Received {len(path_data)} path entries for raw images")
                    logging.info(f"Raw image paths JSON size: {len(raw_image_paths)} characters")
                    if len(path_data) > 0:
                        logging.info(f"Sample path data: {path_data[:3]}")  # Log first 3 entries
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse raw_image_paths JSON: {str(e)}")
                    logging.warning(f"Raw paths data preview: {raw_image_paths[:500]}...")  # Show first 500 chars
            else:
                logging.warning("No raw_image_paths received from frontend")
            
            saved_raw = _get_parents(path_data)
            if raw_image_paths and len(path_data) == 0:
                logging.warning("raw_image_paths is present but contains no valid entries")
            new_raw = saved_raw
            # Clean up and format datetime fields
            formatted_start_time = None if not start_time or start_time.strip() == '' else f"{start_time}:00Z"
            formatted_end_time = None if not end_time or end_time.strip() == '' else f"{end_time}:00Z"
            formatted_condition_time = None if not condition_time or condition_time.strip() == '' else f"{condition_time}:00Z"
            
            # Update text/numeric fields from form input
            updated_metadata = {
                "last_operation": "update",
                "experiment_name": experiment_name,
                "start_time": formatted_start_time,
                "end_time": formatted_end_time,
                "cell_type": cell_type,
                "condition": condition,
                "condition_amount": parsed_condition_amount,
                "condition_unit": condition_unit,
                "condition_time": formatted_condition_time,
                "notes": notes,
                "original_timestamp": existing_metadata.get("timestamp"),
                "last_updated": datetime.now().isoformat() + "Z"
            }
            
            # Handle data files
            if experiment_files and experiment_files[0].filename:
                new_data_files = save_data(experiment_files[0], experiment_name, overwrite=(experiment_files_mode == "replace"))
            
            # Handle overwrites for images
            if any([ac_images_mode == "replace", dc_images_mode == "replace", seg_images_mode == "replace"]):
                exp_processed_dir = os.path.join(IMAGES_DIR, "processed", experiment_name)
                if os.path.exists(exp_processed_dir):
                    for file in os.listdir(exp_processed_dir):
                        should_delete = (
                            (ac_images_mode == "replace" and "ac" in file.lower()) or
                            (dc_images_mode == "replace" and "dc" in file.lower()) or 
                            (seg_images_mode == "replace" and "seg" in file.lower())
                        )
                        if should_delete:
                            os.remove(os.path.join(exp_processed_dir, file))
                            logging.info(f"Deleted {file} due to overwrite mode")
            
            # Handle image files
            existing_images = existing_metadata.get("images", {})
            new_ac = save_processed_images(ac_images, "AC", experiment_name) if ac_images and ac_images[0].filename else []
            new_dc = save_processed_images(dc_images, "DC", experiment_name) if dc_images and dc_images[0].filename else []
            new_seg = save_processed_images(seg_images, "seg", experiment_name) if seg_images and seg_images[0].filename else []
            
            # Apply overwrite logic for each image type
            ac_data = new_ac if (ac_images_mode == "replace" and new_ac) else existing_images.get("ac", []) + new_ac
            dc_data = new_dc if (dc_images_mode == "replace" and new_dc) else existing_images.get("dc", []) + new_dc
            seg_data = new_seg if (seg_images_mode == "replace" and new_seg) else existing_images.get("segmentation", []) + new_seg
            raw_data = new_raw if (raw_images_mode == "replace" and new_raw) else existing_images.get("raw", []) + new_raw
            
            updated_metadata["images"] = {
                "ac": ac_data,
                "dc": dc_data,
                "segmentation": seg_data,
                "raw": raw_data
            }
            
            # Save updated metadata to both data directory and load queue
            save_metadata(updated_metadata)
            meta_filename = f"{experiment_name.replace(' ', '_')}_metadata.json"
            

            # Load experiment into database
            update_experiment_db(os.path.join(DATA_DIR, experiment_name),experiment_name)
            logging.info(f"Experiment '{experiment_name}' uploaded successfully with {1} data files and {len(ac_data) + len(dc_data) + len(seg_data) + len(raw_data)} images.")

            # Create summary message
            has_new_data = bool(experiment_files and experiment_files[0].filename)
            total_new_files = (1 if has_new_data else 0) + len(new_ac) + len(new_dc) + len(new_seg) + len(new_raw)
            msg = f"Experiment '{experiment_name}' updated successfully"
            if total_new_files > 0:
                msg += f" with {total_new_files} new files"
                
                # Add mode information to message
                modes_used = []
                if has_new_data: modes_used.append(f"data ({'replaced' if experiment_files_mode == 'replace' else 'added'})")
                if new_ac: modes_used.append(f"AC images ({'replaced' if ac_images_mode == 'replace' else 'added'})")
                if new_dc: modes_used.append(f"DC images ({'replaced' if dc_images_mode == 'replace' else 'added'})")
                if new_seg: modes_used.append(f"segmentation images ({'replaced' if seg_images_mode == 'replace' else 'added'})")
                if new_raw: modes_used.append(f"raw images ({'replaced' if raw_images_mode == 'replace' else 'added'})")
                
                if modes_used:
                    msg += f" ({', '.join(modes_used)})"
            
            msg += f". Metadata updated in {meta_filename}."
            
            params = urlencode({"message": msg, "success": "1"})
            logging.info(msg)
            return RedirectResponse(url=f"/edit/{experiment_name}?{params}", status_code=303)
            
        except Exception as e:
            logging.error(f"Update failed: {e}")
            params = urlencode({"message": f"Update failed: {str(e)}", "success": "0"})
            return RedirectResponse(url=f"/edit/{experiment_name}?{params}", status_code=303)


@app.delete("/delete/{experiment_name}")
async def delete_experiment_endpoint(experiment_name: str):
    """Delete an experiment and all its associated data"""
    try:
        # Get existing experiments to verify the experiment exists
        experiments = get_existing_experiments()
        
        if experiment_name not in experiments:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
        
        # Use the lock to prevent concurrent operations on the same experiment
        if experiment_name not in experiment_locks:
            experiment_locks[experiment_name] = Lock()
        
        with experiment_locks[experiment_name]:
            # Delete the experiment
            delete_experiment(experiment_name)
            
            # Remove the lock since the experiment no longer exists
            if experiment_name in experiment_locks:
                del experiment_locks[experiment_name]
        
        logging.info(f"Successfully deleted experiment: {experiment_name}")
        return JSONResponse(
            status_code=200,
            content={"message": f"Experiment '{experiment_name}' has been successfully deleted"}
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logging.error(f"Failed to delete experiment '{experiment_name}': {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete experiment: {str(e)}"
        )

@app.get("/download", response_class=HTMLResponse)
def download_experiments_page(request: Request, message: Optional[str] = None, success: Optional[bool] = None):
    """Display page to select experiment for downloading"""
    experiments = get_existing_experiments()
    return templates.TemplateResponse(
        "download_experiments.html",
        {"request": request, "experiments": experiments, "message": message, "success": success}
    )

@app.get("/download/{experiment_name}")
def download_experiment(experiment_name: str):
    """Download an experiment as a ZIP file (data and processed images only, excluding raw images)"""
    import tempfile
    import shutil

    # Get existing experiments to verify the experiment exists
    experiments = get_existing_experiments()
    
    if experiment_name not in experiments:
        params = urlencode({"message": f"Experiment '{experiment_name}' not found", "success": "0"})
        return RedirectResponse(url=f"/download?{params}", status_code=303)
    
    try:
        # Create a temporary file that won't be automatically deleted
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"{experiment_name}_data.zip")
        experiment_dir = Path(DATA_DIR) / experiment_name
        processed_images_dir = Path(IMAGES_DIR)   / "processed" / experiment_name

        # Create ZIP excluding raw images directory
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(experiment_dir):
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, experiment_dir)
                    zip_file.write(file_path, arcname)
            # Add processed images
            for root, dirs, files in os.walk(processed_images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, processed_images_dir)
                    zip_file.write(file_path, os.path.join("processed_images", arcname))

        # Return FileResponse with cleanup callback
        def cleanup():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
        
        return FileResponse(
            path=zip_path,
            filename=f"{experiment_name}_data.zip",
            media_type='application/zip',
            background=cleanup  # This will cleanup after the file is sent
        )
        
    except Exception as e:
        logging.error(f"Failed to create ZIP for experiment '{experiment_name}': {e}")
        params = urlencode({"message": f"Failed to create ZIP: {str(e)}", "success": "0"})
        return RedirectResponse(url=f"/download?{params}", status_code=303)


@app.websocket("/ws/download/{experiment_name}")
async def websocket_download(websocket: WebSocket, experiment_name: str):
    """WebSocket endpoint for downloading large experiments with progress"""
    await websocket.accept()
    
    async def safe_send_json(data):
        """Safely send JSON data, handling disconnections"""
        try:
            await websocket.send_json(data)
            return True
        except Exception as e:
            logging.warning(f"Failed to send WebSocket message: {e}")
            return False
    
    async def safe_send_bytes(data):
        """Safely send bytes data, handling disconnections"""
        try:
            await websocket.send_bytes(data)
            return True
        except Exception as e:
            logging.warning(f"Failed to send WebSocket bytes: {e}")
            return False
    
    try:
        experiments = get_existing_experiments()
        
        if experiment_name not in experiments:
            await safe_send_json({
                "type": "error",
                "message": f"Experiment '{experiment_name}' not found"
            })
            return
        
        # Get experiment directory
        experiment_dir = Path(DATA_DIR) / experiment_name

        # Only include raw images for websocket download
        raw_images_dir = Path(IMAGES_DIR) / "raws" / experiment_name
        
        if not raw_images_dir.exists():
            logging.warning(f"No raw images directory found for experiment '{experiment_name}' at {raw_images_dir}")
            await safe_send_json({
                "type": "error",
                "message": "No raw images found for this experiment"
            })
            return
        
        # Calculate total size for progress (only raw images)
        total_size = sum(f.stat().st_size for f in raw_images_dir.rglob('*') if f.is_file())
        processed_size = 0
        
        # Send start message
        if not await safe_send_json({
            "type": "start",
            "total_size": total_size,
            "filename": f"{experiment_name}_raw_images.zip"
        }):
            return  # Client disconnected
        
        # Create ZIP in chunks and send progress
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
            
            try:
                with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zip_file:
                    for file_path in raw_images_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(raw_images_dir)
                            zip_file.write(file_path, arcname)
                            
                            processed_size += file_path.stat().st_size
                            progress = (processed_size / total_size) * 100 if total_size > 0 else 100
                            
                            # Send progress update
                            if not await safe_send_json({
                                "type": "progress",
                                "progress": progress,
                                "current_file": str(arcname)
                            }):
                                logging.info(f"Client disconnected during ZIP creation for {experiment_name}")
                                return  # Client disconnected, stop processing
                            
                            # Allow other tasks to run
                            await asyncio.sleep(0.01)
                
                # Send the file in chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                bytes_sent = 0
                
                with open(temp_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        
                        # Send chunk
                        if not await safe_send_bytes(chunk):
                            logging.info(f"Client disconnected during file transfer for {experiment_name}")
                            return  # Client disconnected, stop sending
                        
                        bytes_sent += len(chunk)
                        await asyncio.sleep(0.01)  # Prevent overwhelming the connection
                
                # Send completion message
                await safe_send_json({
                    "type": "complete",
                    "message": "Download completed successfully"
                })
                
                logging.info(f"Successfully sent {bytes_sent} bytes for experiment {experiment_name}")
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                    logging.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logging.warning(f"Failed to cleanup temp file {temp_path}: {e}")
                    
    except Exception as e:
        logging.error(f"Download failed for experiment '{experiment_name}': {e}")
        # Try to send error message, but don't fail if websocket is closed
        await safe_send_json({
            "type": "error",
            "message": f"Download failed: {str(e)}"
        })
