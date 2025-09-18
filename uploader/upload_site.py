import os
import logging
import json


from fastapi import FastAPI, UploadFile, File, Request, Form, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional
from file_utils import  DATA_DIR, META_TEMPLATE, save_processed_images, save_raw_images, get_existing_experiments, save_metadata, save_data, delete_experiment, nuke_images
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
):
    if experiment_name not in experiment_locks:
        experiment_locks[experiment_name] = Lock()
    
    with experiment_locks[experiment_name]:
        from datetime import datetime
        try:

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
    """WebSocket endpoint for handling directory uploads"""
    await websocket.accept()
    
    # Get upload_name from query parameters if not provided
    if not upload_name:
        upload_name = websocket.query_params.get('upload_name')
    
    # get overwrite from query parameters
    overwrite_param = websocket.query_params.get('overwrite')
    if overwrite_param is not None:
        overwrite = overwrite_param.lower() in ['true', '1', 'yes']
    
    
    if not upload_name:
        await websocket.send_json({
            "type": "error", 
            "message": "Upload name is required"
        })
        return
    
    await upload_manager.handle_upload(websocket, upload_id, upload_name, overwrite)

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
            nuke_images(exp_name=experiment_name, 
                        ac=(ac_images_mode == "replace"),
                        dc=(dc_images_mode == "replace"),
                        seg=(seg_images_mode == "replace"))
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

    