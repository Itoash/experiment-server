import os
import logging
from fastapi import UploadFile
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import json
import glob
from typing import Dict, Any

# Load environment variables
load_dotenv()
LOAD_QUEUE = os.environ.get("LOAD_QUEUE", "uploads") # Mounted directory for communicating with dataloader; dataloader watches this directory
DATA_DIR = os.environ.get("DATA_PATH", "data")  # Mounted data directory
IMAGES_DIR = os.environ.get("IMAGES_PATH", "images")  # Mounted images directory

# Check if correct directories exist
for directory in [LOAD_QUEUE, DATA_DIR, IMAGES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created missing directory: {directory}")




# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')



def sanitize_filename(filename: str) -> str:
    # Remove path separators and keep only the filename
    return os.path.basename(filename)

def save_data_file(upload_file: UploadFile, metadata: Dict[str, Any], load_queue: str = LOAD_QUEUE, data_dir: str = DATA_DIR, overwrite: bool = False) -> tuple[str, str]:
    """Save uploaded file to the data directory and load queue with metadata
    
    Args:
        upload_file: The uploaded file to save
        metadata: Experiment metadata dictionary
        load_queue: Directory path for dataloader queue
        data_dir: Directory path for permanent data storage
        overwrite: Whether to overwrite existing files
        
    Returns:
        tuple: (filename, metadata_filename)
    """
    
    # Validate file type
    if not upload_file.filename or not upload_file.filename.endswith(('.csv', '.tsv', '.txt', '.json')):
        raise ValueError("Invalid file type. Only .csv, .tsv, .txt, and .json files are allowed.")
    
    # Prepare file paths and names
    exp_name = metadata.get("experiment_name", "unknown_experiment")
    exp_data_dir = os.path.join(data_dir, exp_name)
    os.makedirs(exp_data_dir, exist_ok=True)
    os.makedirs(load_queue, exist_ok=True)
    
    # Create filenames
    sanitized_filename = sanitize_filename(upload_file.filename)
    filename = f'{exp_name}_{sanitized_filename}'
    meta_filename = f"{Path(filename).stem}_metadata.json"
    
    # Resolve filename conflicts if not overwriting
    if not overwrite:
        filename = _resolve_filename_conflicts(filename, exp_data_dir, load_queue)
        meta_filename = f"{Path(exp_name).stem}_metadata.json"
    
    # Create file paths
    data_path = os.path.join(exp_data_dir, filename)
    queue_path = os.path.join(load_queue, filename)
    data_meta_path = os.path.join(exp_data_dir, meta_filename)
    queue_meta_path = os.path.join(load_queue, meta_filename)
    
    # Update metadata with file path
    metadata["data_file"] = data_path
    
    # Read file content once
    upload_file.file.seek(0)  # Reset file pointer to beginning
    file_content = upload_file.file.read()
    
    # Save to both locations
    try:
        # Save data file to permanent storage
        with open(data_path, "wb") as f:
            f.write(file_content)
        logging.info(f"Saved data file: {data_path}")
        
        # Save data file to load queue
        with open(queue_path, "wb") as f:
            f.write(file_content)
        logging.info(f"Saved data file to queue: {queue_path}")
        
        # Save metadata to both locations
        with open(data_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata: {data_meta_path}")
        
        with open(queue_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to queue: {queue_meta_path}")
        
    except Exception as e:
        logging.error(f"Failed to save file {filename}: {e}")
        # Clean up any partially created files
        for path in [data_path, queue_path, data_meta_path, queue_meta_path]:
            if os.path.exists(path):
                os.remove(path)
        raise
    
    return filename, meta_filename

def save_metadata_only(metadata: Dict[str, Any], load_queue: str = LOAD_QUEUE, data_dir: str = DATA_DIR, overwrite: bool = False) -> str:
    """Save only metadata to the data directory and load queue
    
    Args:
        metadata: Experiment metadata dictionary
        load_queue: Directory path for dataloader queue
        data_dir: Directory path for permanent data storage
        overwrite: Whether to overwrite existing files

    Returns:
        str: Metadata filename
    """
    exp_name = metadata.get("experiment_name", "unknown_experiment")
    exp_data_dir = os.path.join(data_dir, exp_name)
    os.makedirs(exp_data_dir, exist_ok=True)
    os.makedirs(load_queue, exist_ok=True)

    meta_filename = f"{exp_name}_metadata.json"
    
    # Resolve filename conflicts if not overwriting
    if not overwrite:
        meta_filename = _resolve_filename_conflicts(meta_filename, exp_data_dir, load_queue)
    
    data_meta_path = os.path.join(exp_data_dir, meta_filename)
    queue_meta_path = os.path.join(load_queue, meta_filename)
    
    try:
        # Save metadata to both locations
        with open(data_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata: {data_meta_path}")
        
        with open(queue_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to queue: {queue_meta_path}")
        
    except Exception as e:
        logging.error(f"Failed to save metadata {meta_filename}: {e}")
        # Clean up any partially created files
        for path in [data_meta_path, queue_meta_path]:
            if os.path.exists(path):
                os.remove(path)
        raise
    print(data_meta_path)
    return meta_filename
def _resolve_filename_conflicts(filename: str, data_dir: str, load_queue: str) -> str:
    """Resolve filename conflicts by appending a counter
    
    Args:
        filename: Original filename
        data_dir: Data directory path
        load_queue: Load queue directory path
        
    Returns:
        str: Non-conflicting filename
    """
    if not (os.path.exists(os.path.join(data_dir, filename)) or 
            os.path.exists(os.path.join(load_queue, filename))):
        return filename
    
    base, ext = os.path.splitext(filename)
    counter = 1
    
    while True:
        new_filename = f"{base}_{counter}{ext}"
        if not (os.path.exists(os.path.join(data_dir, new_filename)) or 
                os.path.exists(os.path.join(load_queue, new_filename))):
            logging.info(f"Filename collision detected. Renamed to {new_filename}")
            return new_filename
        counter += 1




def save_processed_images(files: List[UploadFile], image_type: str, experiment_name: str, image_dir: str= IMAGES_DIR) -> List[str]:
    """Save image files to the images directory organized by experiment and type"""
    if not files or not files[0].filename:
        return []
    
    # Create experiment directory in images
    exp_images_dir = os.path.join(image_dir, "processed", experiment_name)
    os.makedirs(exp_images_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        if file.filename:
            # Save with descriptive filename
            filename = f"{image_type}.tif" if len(files) == 1 else f"{image_type}_{len(saved_files)+1}.tif"
            file_path = os.path.join(exp_images_dir, filename)
            
            with open(file_path, "wb") as buffer:
                while chunk := file.file.read(1024 * 1024):
                    if not chunk:
                        break
                    buffer.write(chunk)
            saved_files.append(filename)
            logging.info(f"Saved image: {file_path}")
    
    return saved_files

def save_raw_images(files: List[UploadFile], experiment_name: str, image_dir: str= IMAGES_DIR) -> List[str]:
    """Save raw image files to the raws directory organized by experiment"""
    if not files or not files[0].filename:
        return []
    
    # Create raw images directory
    raw_dir = os.path.join(image_dir, "raws", experiment_name)
    os.makedirs(raw_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        if file.filename:
            filename = sanitize_filename(file.filename)
            file_path = os.path.join(raw_dir, filename)
            
            with open(file_path, "wb") as buffer:
                while chunk := file.file.read(1024 * 1024):
                    if not chunk:
                        break
                    buffer.write(chunk)
            saved_files.append(filename)
            logging.info(f"Saved raw image: {file_path}")
    
    return saved_files


def get_existing_experiments() -> Dict[str, Any]:
    """Get list of existing experiments from data dir subdirectory names"""
    experiments = {}

    exp_dirs = glob.glob(os.path.join(DATA_DIR, "*/"))
    for exp_dir in exp_dirs:
        experiment_name = os.path.basename(os.path.dirname(exp_dir))
        experiments[experiment_name] = {"data_dir": exp_dir}
        metadata = glob.glob(os.path.join(exp_dir, "*_metadata.json"))
        if metadata:
            with open(metadata[0], "r") as f:
                meta = json.load(f)
                experiments[experiment_name]["metadata"] = meta
        else:
            experiments[experiment_name]["metadata"] = {}
    return experiments
       
