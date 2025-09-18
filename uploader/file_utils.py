import os
import logging
import shutil
from fastapi import UploadFile
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import json
import glob
from typing import Dict, Any
from db_operations import remove_experiment_from_db

# Load environment variables
load_dotenv()
LOAD_QUEUE = os.environ.get("LOAD_QUEUE", "uploads") # Mounted directory for communicating with dataloader; dataloader watches this directory
DATA_DIR = os.environ.get("DATA_PATH", "data")  # Mounted data directory
IMAGES_DIR = os.environ.get("IMAGES_PATH", "images")  # Mounted images directory


# Load metadata template dictionary
with open("meta_template.json", "r") as f:
    META_TEMPLATE = json.load(f)


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

def save_data(upload_file: UploadFile, exp_name:str, data_dir: str = DATA_DIR, load_queue: str = LOAD_QUEUE, overwrite: bool = False) -> str:
    """Save uploaded file to the data directory without metadata
    
    Args:
        upload_file: The uploaded file to save
        exp_name: Experiment name for organizing files
        data_dir: Directory path for permanent data storage
        load_queue: Directory path for dataloader queue
        overwrite: Whether to overwrite existing files
    Returns:
        str: Filename
    """
    # Validate file type
    if not upload_file.filename or not upload_file.filename.endswith(('.csv', '.tsv')):
        raise ValueError("Invalid file type. Only .csv, .tsv files are allowed.")

    # Create filenames
    sanitized_filename = 'cell_data.csv'
    filename = f'{exp_name}_{sanitized_filename}'


    # Prepare file paths and names for data directory
    exp_data_dir = os.path.join(data_dir, exp_name)
    os.makedirs(exp_data_dir, exist_ok=True)
    data_path = os.path.join(exp_data_dir, filename)

   
    
    # Read file content once
    upload_file.file.seek(0)  # Reset file pointer to beginning
    file_content = upload_file.file.read().decode('utf-8')
    
    # Handle CSV/TSV append vs overwrite logic
    try:
        if overwrite or not os.path.exists(data_path):
            # Overwrite mode or file doesn't exist - write new content directly
            final_content = file_content
        else:
            # Append mode - combine existing content with new data rows
            with open(data_path, 'r', encoding='utf-8') as f:
                existing_content = f.read().strip()
            
            if existing_content:
                # Parse new content to separate header from data rows
                new_lines = file_content.strip().split('\n')
                if len(new_lines) > 1:
                    # Do not skip header when appending
                    final_content = existing_content + '\n' + '\n'.join(new_lines)
                else:
                    # New file has only header or is empty
                    final_content = existing_content
            else:
                # Existing file is empty, use new content
                final_content = file_content
        
        # Save final content to both data directory and load queue
        with open(data_path, "w", encoding='utf-8') as f:
            f.write(final_content)
        logging.info(f"Saved data file: {data_path} ({'overwrite' if overwrite else 'append'} mode)")
        
        
        
    except Exception as e:
        logging.error(f"Failed to save file {filename}: {e}")
        # Clean up any partially created files
        for path in [data_path]:
            if os.path.exists(path):
                os.remove(path)
        raise
    
    return data_path



def save_metadata(metadata: Dict[str, Any], load_queue: str = LOAD_QUEUE, data_dir: str = DATA_DIR) -> str:
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

    data_meta_filename = f"{exp_name}_metadata.json"

    data_meta_path = os.path.join(exp_data_dir, data_meta_filename)
    

    try:
        # Save metadata to both locations
        with open(data_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata: {data_meta_path}")
        
    except Exception as e:
        logging.error(f"Failed to save metadata {data_meta_filename}: {e}")
        # Clean up any partially created files
        for path in [data_meta_path]:
            if os.path.exists(path):
                os.remove(path)
        raise
    return data_meta_path


def _resolve_filename_conflicts(filename: str, directory: str) -> str:
    """Resolve filename conflicts by appending a counter
    
    Args:
        filename: Original filename
        directory: Directory path
        
    Returns:
        str: Non-conflicting filename
    """
    if not (os.path.exists(os.path.join(directory, filename))):
        return filename
    
    base, ext = os.path.splitext(filename)
    counter = 1
    
    while True:
        new_filename = f"{base}_{counter}{ext}"
        if not (os.path.exists(os.path.join(directory, new_filename))):
            logging.info(f"Filename collision detected. Renamed to {new_filename}")
            return new_filename
        counter += 1





def save_processed_images(files: List[UploadFile], image_type: str, experiment_name: str, image_dir: str= IMAGES_DIR, overwrite: bool = False) -> List[str]:
    """Save image files to the images directory organized by experiment and type"""
    if not files or not files[0].filename:
        return []
    
    # Create experiment directory in images
    exp_images_dir = os.path.join(image_dir, "processed", experiment_name)
    os.makedirs(exp_images_dir, exist_ok=True)
    
    # Delete existing files of this type if overwrite
    if overwrite:
        for file in os.listdir(exp_images_dir):
            if image_type.lower() in file.lower():
                os.remove(os.path.join(exp_images_dir, file))
                logging.info(f"Removed existing image due to overwrite: {file}")
    saved_files = []
    for file in files:
        if file.filename:
            # Save with descriptive filename
            filename = file.filename
            # Sanitize filename
            filename = sanitize_filename(filename)
            # Resolve conflicts if not overwriting
            filename = _resolve_filename_conflicts(filename, exp_images_dir) if not overwrite else filename
            file_path = os.path.join(exp_images_dir, filename)
            
            with open(file_path, "wb") as buffer:
                while chunk := file.file.read(1024 * 1024):
                    if not chunk:
                        break
                    buffer.write(chunk)
            saved_files.append(filename)
            logging.info(f"Saved image: {file_path}")
    
    return saved_files

def save_raw_images(files: List[UploadFile], experiment_name: str, image_dir: str= IMAGES_DIR, overwrite: bool = False) -> List[str]:
    """Save raw image files to the raws directory organized by experiment, preserving directory structure
    
    Returns:
        List[str]: List of root directory names (for directories) and individual filenames (for files uploaded directly to root).
                  This provides a clean overview of the organizational structure without listing every individual file.
    """
    if not files or not files[0].filename:
        return []
    
    # Create raw images directory
    raw_dir = os.path.join(image_dir, "raws", experiment_name)
    os.makedirs(raw_dir, exist_ok=True)
    
    if overwrite:
        # Clear existing files in the directory and subdirectories
        import shutil
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
            logging.info(f"Removed existing raw directory due to overwrite: {raw_dir}")
        os.makedirs(raw_dir, exist_ok=True)
    
    # Track directories with image files (for clean metadata)
    directories_with_images = set()
    # Track individual files at root level
    root_level_image_files = []
    
    logging.info(f"Processing {len(files)} raw files for experiment: {experiment_name}")
    
    for file in files:
        if file.filename:
            # Check if this is an image file before processing
            is_image = any(file.filename.lower().endswith(ext) for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg'])
            if not is_image:
                logging.info(f"Skipping non-image file: {file.filename}")
                continue  # Skip non-image files
            
            # Use webkitRelativePath if available (from directory upload), otherwise use filename
            if hasattr(file, 'webkitRelativePath') and file.webkitRelativePath:
                relative_path = file.webkitRelativePath
                logging.info(f"Using webkitRelativePath: {relative_path}")
            else:
                # Fallback to filename (may include path separators from manual path construction)
                relative_path = file.filename
                logging.info(f"Using filename as path: {relative_path}")
            
            # Sanitize the path while preserving directory structure
            # Split path and sanitize each component
            path_parts = relative_path.split('/')
            sanitized_parts = [sanitize_filename(part) for part in path_parts if part]
            clean_relative_path = '/'.join(sanitized_parts)
            
            # Create the full file path
            file_path = os.path.join(raw_dir, clean_relative_path)
            
            # Create subdirectories if needed
            file_dir = os.path.dirname(file_path)
            if file_dir != raw_dir:
                os.makedirs(file_dir, exist_ok=True)
                logging.info(f"Created directory: {file_dir}")
            
            # Resolve filename conflicts within the same directory if not overwriting
            if not overwrite and os.path.exists(file_path):
                file_dir = os.path.dirname(file_path)
                filename_only = os.path.basename(file_path)
                resolved_filename = _resolve_filename_conflicts(filename_only, file_dir)
                file_path = os.path.join(file_dir, resolved_filename)
                clean_relative_path = os.path.relpath(file_path, raw_dir)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                file.file.seek(0)  # Reset file pointer
                while chunk := file.file.read(1024 * 1024):
                    if not chunk:
                        break
                    buffer.write(chunk)
            
            logging.info(f"Saved raw image: {file_path}")
            
            # Track directory structure for metadata (only top-level directories)
            if len(sanitized_parts) > 1:
                # This is a file in a subdirectory - track the top-level directory
                root_dir = sanitized_parts[0]
                directories_with_images.add(root_dir)
                logging.info(f"Added directory to metadata: {root_dir}")
            else:
                # This is a file directly at the root level
                root_level_image_files.append(sanitized_parts[0])
                logging.info(f"Added root-level file to metadata: {sanitized_parts[0]}")
    
    # Return ONLY the top-level directory names and any individual files at root
    # This keeps metadata clean and prevents pollution with individual filenames
    result = list(directories_with_images) + root_level_image_files
    logging.info(f"Returning metadata entries: {result}")
    return result


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


def delete_experiment(experiment_name: str, data_dir: str = DATA_DIR, image_dir: str = IMAGES_DIR) -> None:
    """Delete all files associated with an experiment"""
    exp_data_dir = os.path.join(data_dir, experiment_name)
    exp_raw_image_dir = os.path.join(image_dir, "raws", experiment_name)
    exp_processed_image_dir = os.path.join(image_dir, "processed", experiment_name)
    
    for path in [exp_data_dir, exp_raw_image_dir, exp_processed_image_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"Deleted directory: {path}")
        else:
            logging.warning(f"Directory not found, cannot delete: {path}")

    remove_experiment_from_db(experiment_name)
