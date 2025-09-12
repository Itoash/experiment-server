import os
import logging
import json
import glob
from fastapi import FastAPI, UploadFile, File, Request, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
from file_utils import  DATA_DIR, IMAGES_DIR, LOAD_QUEUE, save_processed_images, save_raw_images, get_existing_experiments, save_data_file, save_metadata_only
from urllib.parse import urlencode
from dotenv import load_dotenv 
load_dotenv()  


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/", response_class=HTMLResponse)
def main(request: Request, message: Optional[str] = None, success: Optional[bool] = None):
    return templates.TemplateResponse(
        "upload_form.html",
        {"request": request, "message": message, "success": success}
    )

@app.get("/safari-test", response_class=HTMLResponse)
def safari_test(request: Request):
    """Test page for Safari compatibility"""
    return templates.TemplateResponse(
        "safari_test.html",
        {"request": request}
    )

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
    raw_images: List[UploadFile] = File(default=[])
):
    from datetime import datetime
    try:
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

        # Save raw images to raw directory
        saved_raw = save_raw_images(raw_images, experiment_name)
        
        # Save metadata JSON with all form data
        metadata = {
            "last_operation": "create",
            "experiment_name": experiment_name,
            "start_time": start_time,
            "end_time": end_time,
            "cell_type": cell_type,
            "condition": condition,
            "condition_amount": parsed_condition_amount,
            "condition_unit": condition_unit,
            "condition_time": condition_time,
            "notes": notes,
            "images": {
                "ac": saved_ac,
                "dc": saved_dc,
                "segmentation": saved_seg,
                "raw": saved_raw
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        # Save experiment data files to uploads directory
        saved_data_files, meta_filename = save_data_file(experiment_files[0], metadata)

        # Create summary message
        total_images = len(saved_ac) + len(saved_dc) + len(saved_seg) + len(saved_raw)
        msg = f"Experiment '{experiment_name}' uploaded: {len(saved_data_files)} data files, {total_images} images. Metadata saved as {meta_filename}."
        params = urlencode({"message": msg, "success": "1"})
        return RedirectResponse(url=f"/?{params}", status_code=303)

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        params = urlencode({"message": f"Upload failed: {str(e)}", "success": "0"})
        return RedirectResponse(url=f"/?{params}", status_code=303)


    
    return experiments

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
    raw_images: List[UploadFile] = File(default=[]),
    raw_images_mode: Optional[str] = Form(None)  # Add mode toggle
):
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
        
        # Update text/numeric fields from form input
        updated_metadata = {
            "last_operation": "update",
            "experiment_name": experiment_name,
            "start_time": start_time,
            "end_time": end_time,
            "cell_type": cell_type,
            "condition": condition,
            "condition_amount": parsed_condition_amount,
            "condition_unit": condition_unit,
            "condition_time": condition_time,
            "notes": notes,
            "original_timestamp": existing_metadata.get("timestamp"),
            "last_updated": datetime.now().isoformat() + "Z"
        }
        
        # Handle data files
        existing_data_files = existing_metadata.get("data_files", [])
        if experiment_files and experiment_files[0].filename:
            # Create metadata for new data file
            data_file_metadata = updated_metadata.copy()
            new_data_filename, _ = save_data_file(experiment_files[0], data_file_metadata)
            
            if experiment_files_mode == "replace":
                updated_metadata["data_files"] = [new_data_filename]
            else:
                updated_metadata["data_files"] = existing_data_files + [new_data_filename]
        else:
            # No new data file, preserve existing data files
            updated_metadata["data_files"] = existing_data_files
        
        # Handle image files
        existing_images = existing_metadata.get("images", {})
        new_ac = save_processed_images(ac_images, "AC", experiment_name) if ac_images and ac_images[0].filename else []
        new_dc = save_processed_images(dc_images, "DC", experiment_name) if dc_images and dc_images[0].filename else []
        new_seg = save_processed_images(seg_images, "seg", experiment_name) if seg_images and seg_images[0].filename else []
        new_raw = save_raw_images(raw_images, experiment_name) if raw_images and raw_images[0].filename else []
        
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
        save_metadata_only(updated_metadata, overwrite=True)
        meta_filename = f"{experiment_name.replace(' ', '_')}_metadata.json"
        
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
        return RedirectResponse(url=f"/edit/{experiment_name}?{params}", status_code=303)
        
    except Exception as e:
        logging.error(f"Update failed: {e}")
        params = urlencode({"message": f"Update failed: {str(e)}", "success": "0"})
        return RedirectResponse(url=f"/edit/{experiment_name}?{params}", status_code=303)
    
    