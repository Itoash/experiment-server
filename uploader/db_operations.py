import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import json
from typing import List
from datetime import datetime
import logging


DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://researcher:biodinamica2025!@db:5432/experiment_db')
IMAGE_DIR = os.getenv('IMAGE_DIR', '/mnt/images')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    
def read_cell_data(path):
    """Reads cell data from a csv/tsv file and returns a list of dictionaries, each representing a cell."""

    extension = os.path.splitext(path)[1]
    if extension not in ['.csv','.tsv','.txt']:
        raise ValueError('File must be a .csv, .tsv, or .txt')
    if extension == '.csv':
        delimiter = ','
    elif extension == '.tsv':
        delimiter = '\t'
    else:
        delimiter = ','
    cells = []
    with open(path) as file:
        lines = file.readlines()
        
    cell_starts = []
    for i,line in enumerate(lines):
        if 'cell' in line.lower():
            cell_starts.append(i)
    
    for idx in cell_starts:
        celldict = {}
        for line in lines[idx+1:idx+21]:
            strlist = line.strip().split(delimiter)
            name = strlist[0]
            if 'position' in name:
                try: 
                    position_x = [float(strlist[i].lstrip(' ["')) for i in range(1,len(strlist),2)]
                    position_y = [float(strlist[i].rstrip(' ]"')) for i in range(2,len(strlist),2)]
                    celldict.update({'X':position_x})
                    celldict.update({'Y':position_y})
                except IndexError:
                    print(path)
                    print(strlist)
            else:
                try:
                    data = [float(x.strip()) for x in strlist[1:]]
                    
                except:
                    logging.info(f'Error at attribue {name}')
            celldict.update({name:np.array(data)})
        celldict.update({'name':lines[idx].strip()})
        if len(celldict['name'].split('_'))>2:
            # Means cell is a daughter cell
            celldict.update({'parent': '_'.join(celldict['name'].split('_')[:-1])})
        else:
            celldict.update({'parent': None})
        cells.append(celldict)
    return cells

def read_timepoints(meta_path:str,csv_path:str):
    """Reads timepoint data from a text file and returns a list of dictionaries, each representing a timepoint.
    timepoints: List of timestamps in date-time format.
    intervals: List of time intervals in minutes since start of experiment.
    """
    with open(meta_path) as file:
        metadata = json.load(file)
    exp_name = metadata.get('experiment_name', 'unknown_experiment')

    with open(csv_path, 'r') as file:
        lines = file.readlines()
        times = []
        for line in lines:
            if 'times' in line:
                times+= line.strip().split(',')[1:]
        times = [float(t) for t in times]
        times = list(set(times))
        times = list(sorted(times))
    start_time = metadata.get('start_time', None)
    if start_time:
        intervals = times.copy()
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp()
        times = [start_time+t*60 for t in times]
    return times, intervals

def read_image_paths(meta_path:str):
    """Reads image paths from metadata and returns structured data about images."""
    with open(meta_path) as file:
        metadata = json.load(file)
    exp_name = metadata.get('experiment_name', 'unknown_experiment')
    
    # Look for processed images in the expected locations
    processed_images_base = os.path.join(IMAGE_DIR, 'processed', exp_name)
    raw_images_base = os.path.join(IMAGE_DIR, 'raws', exp_name)

    # Check for processed images
    AC_path = []
    DC_path = []
    seg_path = []
    
    if os.path.exists(processed_images_base):
        for f in os.listdir(processed_images_base):
            if f.lower().endswith('.tif'):
                if 'ac' in f.lower():
                    AC_path.append(os.path.join(processed_images_base, f))
                elif 'dc' in f.lower():
                    DC_path.append(os.path.join(processed_images_base, f))
                elif 'seg' in f.lower():
                    seg_path.append(os.path.join(processed_images_base, f))
    
    # Check for raw image directories
    raws_paths = []
    if os.path.exists(raw_images_base):
        for item in os.listdir(raw_images_base):
            item_path = os.path.join(raw_images_base, item)
            if os.path.isdir(item_path):
                raws_paths.append(item_path)

    image_data = {
        'AC': AC_path,
        'DC': DC_path, 
        'segmentation': seg_path,
        'raw': raws_paths,
        'metadata': metadata
    }
    logging.info(f'Found image data: { image_data}')
    return image_data

def parse_raw_paths(raws: List[str], timepoints: List[float]) -> List[int]:
    """Extracts timepoint indices from raw image file paths/metadata.
    Returns a list of integers representing timepoint indices.
    Expected input for raws is a directory with raw images. The directory name follows the format:
    100x_DICTL5v7_926mV_1Hz_40ms_16v8fps_YYYYMMDD_HHMMSS AM/PM
    """
    timepoint_indices = []
    
    for raw in raws:
        if os.path.isdir(raw):
            dir_name = os.path.basename(raw)
            # Parse the directory name format: 100x_DICTL5v7_926mV_1Hz_40ms_16v8fps_20241104_42145 PM
            parts = dir_name.split('_')
            
            # Find date and time parts
            date_part = None
            time_part = None
            am_pm = None
            
            for i, part in enumerate(parts):
                # Look for 8-digit date (YYYYMMDD)
                if len(part) == 8 and part.isdigit():
                    date_part = part
                    # Time should be next part
                    if i + 1 < len(parts):
                        time_part = parts[i + 1][:-2].strip() # Remove AM/PM if attached
                    # AM/PM should be after that or separate
                    if i + 2 < len(parts):
                        am_pm = parts[i + 2].strip()
                    elif ' ' in dir_name:
                        # AM/PM might be separated by space
                        am_pm_split = dir_name.split(' ')
                        if len(am_pm_split) > 1:
                            am_pm = am_pm_split[-1].strip()
                    break
            logging.info(f'Parsing directory: {dir_name}, found date: {date_part}, time: {time_part}, am/pm: {am_pm}')
            
            if date_part and time_part:
                try:
                    # Parse date: 20241104 -> 2024-11-04
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    
                    # Parse time: 42145 -> 04:21:45 (assuming HMMSS format)
                    if len(time_part) == 5:
                        hour = time_part[0]
                        minute = time_part[1:3]
                        second = time_part[3:5]
                    elif len(time_part) == 6:
                        hour = time_part[:2]
                        minute = time_part[2:4]
                        second = time_part[4:6]
                    else:
                        logging.warning(f"Could not parse time format: {time_part}")
                        continue
                    
                    # Create datetime string
                    datetime_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                    if am_pm and am_pm.upper() in ['AM', 'PM']:
                        datetime_str += f" {am_pm.upper()}"
                        dt = datetime.strptime(datetime_str, '%Y-%m-%d %I:%M:%S %p')
                    else:
                        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                    
                    timestamp = dt.timestamp()
                    logging.info(f'Parsed timestamp: {timestamp} for directory: {dir_name}')
                    logging.info(f'Available timepoints: {timepoints}')
                    # Find the closest timepoint
                    if timepoints:
                        time_diffs = [abs(tp - timestamp) for tp in timepoints]
                        closest_index = time_diffs.index(min(time_diffs))
                        timepoint_indices.append(closest_index)
                    else:
                        timepoint_indices.append(0)  # Default to first timepoint
                        
                except (ValueError, IndexError) as e:
                    logging.warning(f"Could not parse date/time from directory name: {dir_name}, error: {e}")
                    timepoint_indices.append(0)  # Default to first timepoint
            else:
                logging.warning(f"Could not find date/time parts in directory name: {dir_name}")
                timepoint_indices.append(0)  # Default to first timepoint
                
        elif os.path.isfile(raw) and raw.endswith('.sif'):
            # Handle Andor spool files
            try:
                file_creation_time = os.path.getctime(raw)
                if timepoints:
                    time_diffs = [abs(tp - file_creation_time) for tp in timepoints]
                    closest_index = time_diffs.index(min(time_diffs))
                    timepoint_indices.append(closest_index)
                else:
                    timepoint_indices.append(0)
            except Exception as e:
                print(f"Could not get creation time for file: {raw}, error: {e}")
                timepoint_indices.append(0)
        else:
            print(f"Unknown raw data format: {raw}")
            timepoint_indices.append(0)
    
    return timepoint_indices


# Data conversion methods to create SQLAlchemy model objects

def create_experiment_from_metadata(metadata_path: str):
    """Creates an Experiment object from metadata JSON file."""
    from models_db import Experiment
    
    session = SessionLocal()
    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        
        # Check if experiment already exists
        experiment_name = metadata.get('experiment_name', 'unknown_experiment')
        assert(metadata['experiment_name'] == experiment_name)  # Ensure name is in metadata
        existing_experiment = session.query(Experiment).filter_by(name=experiment_name).first()
        if existing_experiment:
            raise ValueError(f"Experiment with name '{experiment_name}' already exists")
        
        # Parse dates
        start_time = None
        end_time = None
        
        if 'start_time' in metadata:
            start_time = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
        elif 'date' in metadata and metadata['date']:
            start_time = datetime.fromisoformat(metadata['date'].replace('Z', '+00:00'))
        
        if 'end_time' in metadata and metadata['end_time']:
            end_time = datetime.fromisoformat(metadata['end_time'].replace('Z', '+00:00'))
        
        experiment = Experiment(
            name=metadata.get('experiment_name', 'unknown_experiment'),
            start_time=start_time,
            end_time=end_time,
            cell_type=metadata.get('cell_type'),
            condition=metadata.get('condition'),
            condition_amount=metadata.get('condition_amount'),
            condition_unit=metadata.get('condition_unit'),
            condition_time=metadata.get('condition_time'),
            notes=metadata.get('notes'),
            metadata=metadata   
        )
        
        return experiment
        
    finally:
        session.close()

def create_timepoints_from_data(experiment_id: int, times: List[float], intervals: List[float]):
    """Creates Timepoint objects from timepoint data."""
    from models_db import Timepoint
    from datetime import timedelta
    
    timepoints = []
    
    for i, (time, interval) in enumerate(zip(times, intervals)):
        # Convert timestamp to datetime if it's a timestamp
        if isinstance(time, (int, float)):
            timestamp_absolute = datetime.fromtimestamp(time)
        else:
            timestamp_absolute = time
        
        # Convert interval to timedelta
        timestamp_relative = timedelta(minutes=interval) if interval is not None else None
        
        timepoint = Timepoint(
            experiment_id=experiment_id,
            timepoint_number=i,
            timestamp_absolute=timestamp_absolute,
            timestamp_relative=timestamp_relative
        )
        timepoints.append(timepoint)
    
    return timepoints

def create_cells_from_data(experiment_id: int, cells_data: List[dict], session):
    """Creates Cell objects from cell data."""
    from models_db import Cell
    
    cells = []
    cell_name_to_id = {}  # To map cell names to database IDs later
    
    for cell_dict in cells_data:
        cell_name = cell_dict.get('name', '')
        parent_name = cell_dict.get('parent')
        
        cell = Cell(
            experiment_id=experiment_id,
            cell_identifier=cell_name,
            parent_cell_id=None,  # Will be set after all cells are created
            generation=0  # Will be calculated by trigger
        )
        
        cells.append(cell)
        session.add(cell)
        session.flush()  # Get cell ID
        cell_name_to_id[cell_name] = cell.id
    
    return cells, cell_name_to_id

def create_observations_from_cell_data(experiment_id: int, cells_data: List[dict], 
                                     timepoints: List, observable_types: dict):
    """Creates CellObservation objects from cell measurement data."""
    from models_db import CellObservation
    
    observations = []
    
    for cell_dict in cells_data:
        cell_name = cell_dict.get('name', '')
        
        # Process each measurement type
        for measurement_name, values in cell_dict.items():
            if measurement_name in ['name', 'parent', 'times', 'idx']:
                continue
            
            # Handle position data specially
            if measurement_name == 'X' or measurement_name == 'Y':
                obs_type_name = f'position_{measurement_name.lower()}'
            else:
                obs_type_name = measurement_name
            
            # Get observable type ID
            if obs_type_name not in observable_types:
                print(f"Warning: Observable type '{obs_type_name}' not found in database")
                continue
            
            observable_type_id = observable_types[obs_type_name].id
            
            # Create observations for each timepoint
            if hasattr(values, '__len__') and not isinstance(values, str):
                for i, value in enumerate(values):
                    if i < len(timepoints) and timepoints[i] is not None and hasattr(timepoints[i], 'id') and timepoints[i].id is not None:
                        # Skip NaN values
                        if isinstance(value, float) and np.isnan(value):
                            continue
                        observation = CellObservation(
                            experiment_id=experiment_id,
                            timepoint_id=timepoints[i].id,
                            cell_id=None,  # Will be set after cell is saved
                            observable_type_id=observable_type_id,
                            value_numeric=float(value) if isinstance(value, (int, float)) else None,
                            value_text=str(value) if not isinstance(value, (int, float)) else None
                        )
                        observations.append((observation, cell_name))  # Store with cell name for later linking
    
    return observations

def create_processed_images_from_data(experiment_id: int, image_data: dict):
    """Creates ProcessedImage objects from image path data."""
    from models_db import ProcessedImage
    
    processed_images = []
    
    image_types = {
        'AC': image_data.get('AC'),
        'DC': image_data.get('DC'),
        'Segmentation': image_data.get('segmentation')
    }
    
    for image_type, file_paths in image_types.items():
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                processed_image = ProcessedImage(
                    experiment_id=experiment_id,
                    file_path=file_path,
                    image_type=image_type
                )
                processed_images.append(processed_image)
    
    return processed_images

def create_raw_images_from_data(experiment_id: int, raw_paths: List[str], 
                               timepoint_indices: List[int], timepoints: List):
    """Creates RawImage objects from raw image paths and timepoint indices."""
    from models_db import RawImage
    logging.info(f'Creating raw images for experiment {experiment_id}')
    logging.info(f'Raw paths: {raw_paths}')
    logging.info(f'Timepoint indices: {timepoint_indices}')
    raw_images = []
    
    for raw_path, timepoint_idx in zip(raw_paths, timepoint_indices):
        if timepoint_idx < len(timepoints):
            timepoint_id = timepoints[timepoint_idx].id
            if os.path.isdir(raw_path):
            # Add the directory path as is
                raw_image = RawImage(
                    experiment_id=experiment_id,
                    file_path=raw_path,
                    timepoint_id=timepoint_id
                )
                raw_images.append(raw_image)
            logging.info(f'Added raw image: {raw_path} for timepoint index {timepoint_idx}')
                
    
    return raw_images


# Helper methods for updating existing experiments
def _set_parent_cell_ids(cells, cell_name_to_id):
    """Sets parent_cell_id for each Cell object based on cell_name_to_id mapping."""
    for cell in cells:
        if cell.parent_cell_id is None and cell.cell_identifier in cell_name_to_id:
            parent_name = cell.cell_identifier.rsplit('_', 1)[0]  # Assuming parent name is prefix before last '_'
            if parent_name in cell_name_to_id:
                cell.parent_cell_id = cell_name_to_id[parent_name]
    return cells

def _set_birth_death_times(cells, cells_data, timepoints):
    """Sets birth_time and death_time for each Cell object based on cells_data and timepoints."""
    timepoint_map = {tp.timepoint_number: tp for tp in timepoints}
    
    for cell, cell_dict in zip(cells, cells_data):
        # Get first and last indices
        first_idx = cell_dict['idx'][0] 
        last_idx = cell_dict['idx'][-1]
        # Get timepoint id for birth time
        if first_idx in timepoint_map:
            cell.birth_timepoint_id = timepoint_map[first_idx].id
        if last_idx in timepoint_map:
            cell.death_timepoint_id = timepoint_map[last_idx].id

    return cells

def _update_experiment_metadata(experiment, metadata, session):
    """Updates experiment metadata fields."""
    # Always update these fields if they exist in metadata
    if 'notes' in metadata:
        experiment.notes = metadata['notes']
    if 'start_time' in metadata:
        experiment.start_time = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
    if 'end_time' in metadata and metadata['end_time']:
        experiment.end_time = datetime.fromisoformat(metadata['end_time'].replace('Z', '+00:00'))
    if 'cell_type' in metadata:
        experiment.cell_type = metadata['cell_type']
    if 'condition' in metadata:
        experiment.condition = metadata['condition']
    if 'condition_amount' in metadata:
        experiment.condition_amount = metadata['condition_amount']
    if 'condition_unit' in metadata:
        experiment.condition_unit = metadata['condition_unit']
    if 'condition_time' in metadata:
        experiment.condition_time = metadata['condition_time']
    
    # Always update the full metadata JSON
    experiment.metadata = metadata
    
    # Mark the experiment as dirty to ensure it gets updated
    session.add(experiment)

def _update_or_create_timepoints(experiment_id: int, times: List[float], intervals: List[float], session):
    """Updates existing timepoints by deleting old ones and creating new ones."""
    from models_db import Timepoint, Cell, CellObservation, RawImage
    from datetime import timedelta
    
    # First, delete all related data that references timepoints
    # Delete observations first (they reference both cells and timepoints)
    session.query(CellObservation).filter_by(experiment_id=experiment_id).delete()
    
    # Delete raw images that reference timepoints
    session.query(RawImage).filter_by(experiment_id=experiment_id).delete()
    
    # Update cells to remove timepoint references
    cells = session.query(Cell).filter_by(experiment_id=experiment_id).all()
    for cell in cells:
        cell.birth_timepoint_id = None
        cell.death_timepoint_id = None
    session.flush()
    
    # Now we can safely delete timepoints
    deleted_count = session.query(Timepoint).filter_by(experiment_id=experiment_id).delete()
    logging.info(f'Deleted {deleted_count} existing timepoints')
    
    # Create new timepoints
    timepoints = create_timepoints_from_data(experiment_id, times, intervals)
    session.add_all(timepoints)
    session.flush()  # Get timepoint IDs
    
    logging.info(f'Created {len(timepoints)} new timepoints')
    return timepoints

def _update_or_create_cells(experiment_id: int, cells_data: List[dict], timepoints: List, session):
    """Updates existing cells by deleting old ones and creating new ones."""
    from models_db import Cell, CellObservation
    
    # Delete observations that reference cells (if not already deleted)
    # This is safe to call multiple times
    session.query(CellObservation).filter_by(experiment_id=experiment_id).delete()
    
    # Delete existing cells (timepoint references should already be cleared)
    deleted_count = session.query(Cell).filter_by(experiment_id=experiment_id).delete()
    logging.info(f'Deleted {deleted_count} existing cells')
    
    # Create new cells and get their ids
    cells, cell_name_to_id = create_cells_from_data(experiment_id, cells_data, session)
    
    # Set parent relationships and birth/death times
    cells = _set_parent_cell_ids(cells, cell_name_to_id)
    
    # Set birth/death times using new timepoints
    timepoint_map = {tp.timepoint_number: tp for tp in timepoints}
    
    for cell, cell_dict in zip(cells, cells_data):
        # Set birth/death times by finding first and last timepoints with data
        # Look through all measurement arrays to find valid timepoint indices
        valid_indices = set()
        for measurement_name, values in cell_dict.items():
            if measurement_name not in ['name', 'parent'] and hasattr(values, '__len__') and not isinstance(values, str):
                for i, value in enumerate(values):
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        valid_indices.add(i)
        
        if valid_indices:
            first_idx = min(valid_indices)
            last_idx = max(valid_indices)
            
            if first_idx in timepoint_map:
                cell.birth_timepoint_id = timepoint_map[first_idx].id
            if last_idx in timepoint_map:
                cell.death_timepoint_id = timepoint_map[last_idx].id
    
    session.add_all(cells)  # Update cells with parent IDs and timepoint references
    session.flush()

    logging.info(f'Created {len(cells)} new cells')
    return cells, cell_name_to_id

def _update_cell_observations(experiment_id: int, cells_data: List[dict], timepoints: List, 
                             observable_types: dict, cells: List, session):
    """Updates cell observations by deleting old ones and creating new ones."""
    from models_db import CellObservation
    
    # Validate timepoints before proceeding
    valid_timepoints = [tp for tp in timepoints if tp is not None and hasattr(tp, 'id') and tp.id is not None]
    logging.info(f'Valid timepoints: {len(valid_timepoints)} out of {len(timepoints)}')
    
    if len(valid_timepoints) == 0:
        logging.error('No valid timepoints found - cannot create observations')
        return
    
    # Delete existing observations for this experiment (if not already deleted)
    # This is safe to call multiple times
    deleted_count = session.query(CellObservation).filter_by(experiment_id=experiment_id).delete()
    session.flush()
    logging.info(f'Deleted {deleted_count} existing observations (if any remaining)')
    
    # Create new observations (reuse existing logic)
    observations_with_cell_names = create_observations_from_cell_data(
        experiment_id, cells_data, valid_timepoints, observable_types
    )
    
    # Link observations to cells
    cell_name_to_obj = {cell.cell_identifier: cell for cell in cells}
    observations = []
    for obs, cell_name in observations_with_cell_names:
        if cell_name in cell_name_to_obj:
            obs.cell_id = cell_name_to_obj[cell_name].id
            observations.append(obs)
    
    logging.info(f'Creating {len(observations)} new observations')
    session.add_all(observations)

def _update_processed_images(experiment_id: int, image_data: dict, session):
    """Updates processed images by replacing existing ones."""
    from models_db import ProcessedImage
    
    # Delete existing processed images
    deleted_count = session.query(ProcessedImage).filter_by(experiment_id=experiment_id).delete()
    session.flush()
    logging.info(f'Deleted {deleted_count} existing processed images')
    
    # Create new processed images (reuse existing logic)
    processed_images = create_processed_images_from_data(experiment_id, image_data)
    if processed_images:
        session.add_all(processed_images)
        logging.info(f'Created {len(processed_images)} new processed images')

def _update_raw_images(experiment_id: int, raw_paths: List[str], 
                      timepoint_indices: List[int], timepoints: List, session):
    """Updates raw images by replacing existing ones."""
    from models_db import RawImage
    
    # Delete existing raw images (if not already deleted)
    # This is safe to call multiple times
    deleted_count = session.query(RawImage).filter_by(experiment_id=experiment_id).delete()
    session.flush()
    logging.info(f'Deleted {deleted_count} existing raw images (if any remaining)')
    
    # Create new raw images (reuse existing logic)
    raw_images = create_raw_images_from_data(experiment_id, raw_paths, timepoint_indices, timepoints)
    if raw_images:
        session.add_all(raw_images)
        logging.info(f'Created {len(raw_images)} new raw images')

def load_experiment_complete(data_path: str):
    """Complete pipeline to load experiment data and create all SQLAlchemy objects."""
    from models_db import ObservableType
    from sqlalchemy.exc import IntegrityError
    
    session = SessionLocal()
    try:
        # Read all data
        meta_name = [f for f in os.listdir(data_path) if 'metadata.json' in f.lower()]
        if len(meta_name) == 0:
            raise ValueError('No metadata.json file found in the provided data path')
        else:
            meta_name = meta_name[0]
        csv_name = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_name) == 0:
            raise ValueError('No CSV file found in the provided data path')
        else:
            csv_name = csv_name[0]

        meta_path = os.path.join(data_path, meta_name)
        csv_path = os.path.join(data_path, csv_name)
        if not os.path.exists(meta_path):
            raise ValueError(f'Metadata file not found at {meta_path}')
        if not os.path.exists(csv_path):
            raise ValueError(f'CSV file not found at {csv_path}')
        
        cells_data = read_cell_data(csv_path)
        times, intervals = read_timepoints(meta_path, csv_path)
        image_data = read_image_paths(meta_path)
        logging.info(f'Image data found: {image_data}')
        logging.info(f'Number of cells read: {len(cells_data)}')
        logging.info(f'Number of timepoints read: {len(times)}')

        try:
            # Create experiment
            experiment = create_experiment_from_metadata(meta_path)
            session.add(experiment)
            session.flush()  # Get the experiment ID
        except IntegrityError:
            session.rollback()
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                experiment_name = metadata.get('experiment_name', 'unknown_experiment')
                raise ValueError(f"Cannot load experiment: an experiment with name '{experiment_name}' already exists")
        except ValueError as e:
            session.rollback()
            raise e
        
        # Create timepoints
        timepoints = create_timepoints_from_data(experiment.id, times, intervals)
        session.add_all(timepoints)
        session.flush()  # Get timepoint IDs
        
        # Create cells and get their IDs
        cells, cell_name_to_id = create_cells_from_data(experiment.id, cells_data, session)
        db_cells = session.query(cells[0].__class__).filter_by(experiment_id=experiment.id).all()
        print(f'Number of cells in DB: {len(db_cells)}')
        cells = _set_parent_cell_ids(cells, cell_name_to_id)
        cells = _set_birth_death_times(cells, cells_data, timepoints)
        session.add_all(cells)  # Update cells with parent IDs
        session.flush()
        
        # Get observable types from database
        observable_types = {ot.name: ot for ot in session.query(ObservableType).all()}
        
        # Create observations
        observations_with_cell_names = create_observations_from_cell_data(
            experiment.id, cells_data, timepoints, observable_types
        )
        
        # Link observations to cells
        cell_name_to_obj = {cell.cell_identifier: cell for cell in cells}
        observations = []
        for obs, cell_name in observations_with_cell_names:
            if cell_name in cell_name_to_obj:
                obs.cell_id = cell_name_to_obj[cell_name].id
                observations.append(obs)
        
        session.add_all(observations)
        
        # Create processed images
        processed_images = create_processed_images_from_data(experiment.id, image_data)
        session.add_all(processed_images)
        
        # Create raw images
        if image_data.get('raw'):
            timepoint_indices = parse_raw_paths(image_data['raw'], times)
            raw_images = create_raw_images_from_data(
                experiment.id, image_data['raw'], timepoint_indices, timepoints
            )
            session.add_all(raw_images)
        
        # Commit all changes
        session.commit()
        logging.info(f'Successfully loaded experiment: {experiment.name}')
        return experiment
        
    except Exception as e:
        session.rollback()
        logging.error(f'Error loading experiment: {e}')
        raise e
    finally:
        session.close()

def update_experiment(data_path: str, experiment_name: str = None):
    """Updates an existing experiment with new data.
        For now, delete and regenerate all data
        """
    from models_db import Experiment, ObservableType, Cell, Timepoint, CellObservation, ProcessedImage, RawImage
    from sqlalchemy.exc import IntegrityError
    
    session = SessionLocal()
    try:
        logging.info(f'=== UPDATE_EXPERIMENT CALLED with data_path: {data_path} ===')
        # Read all data files
        meta_name = [f for f in os.listdir(data_path) if 'metadata.json' in f.lower()]
        if len(meta_name) == 0:
            raise ValueError('No metadata.json file found in the provided data path')
        else:
            meta_name = meta_name[0]
        
        csv_name = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_name) == 0:
            raise ValueError('No CSV file found in the provided data path')
        else:
            csv_name = csv_name[0]

        meta_path = os.path.join(data_path, meta_name)
        csv_path = os.path.join(data_path, csv_name)
        
        if not os.path.exists(meta_path):
            raise ValueError(f'Metadata file not found at {meta_path}')
        if not os.path.exists(csv_path):
            raise ValueError(f'CSV file not found at {csv_path}')
        
        # Read metadata to identify experiment
        with open(meta_path, 'r') as file:
            metadata = json.load(file)
        
        experiment_name = metadata.get('experiment_name', 'unknown_experiment')
        existing_experiment = session.query(Experiment).filter_by(name=experiment_name).first()
        if not existing_experiment:
            raise ValueError(f"Experiment with name '{experiment_name}' does not exist for update")
        
        logging.info(f'Updating experiment: {experiment_name}')
        
        # Read all new data
        cells_data = read_cell_data(csv_path)
        times, intervals = read_timepoints(meta_path, csv_path)
        image_data = read_image_paths(meta_path)
        
        logging.info(f'Number of cells to update: {len(cells_data)}')
        logging.info(f'Number of timepoints to update: {len(times)}')
        
        # Update experiment metadata first
        _update_experiment_metadata(existing_experiment, metadata, session)
        session.flush()
        logging.info(f'Updated experiment metadata')
        
        # Update timepoints (this will cascade delete related data in the correct order)
        existing_timepoints = _update_or_create_timepoints(existing_experiment.id, times, intervals, session)
        session.flush()
        logging.info(f'Updated {len(existing_timepoints)} timepoints')
        
        # Update cells (observations and raw images already deleted by timepoint update)
        existing_cells, cell_name_to_id = _update_or_create_cells(existing_experiment.id, cells_data, existing_timepoints, session)
        session.flush()
        logging.info(f'Updated {len(existing_cells)} cells')
        
        # Get observable types from database
        observable_types = {ot.name: ot for ot in session.query(ObservableType).all()}
        
        # Update observations (create new ones)
        _update_cell_observations(existing_experiment.id, cells_data, existing_timepoints, observable_types, existing_cells, session)
        session.flush()
        logging.info(f'Updated cell observations')
        
        # Update processed images
        _update_processed_images(existing_experiment.id, image_data, session)
        session.flush()
        logging.info(f'Updated processed images')
        
        # Update raw images
        if image_data.get('raw'):
            timepoint_indices = parse_raw_paths(image_data['raw'], times)
            _update_raw_images(existing_experiment.id, image_data['raw'], timepoint_indices, existing_timepoints, session)
            session.flush()
            logging.info(f'Updated raw images')
        
        # Commit all changes
        session.commit()
        logging.info(f'Successfully updated experiment: {experiment_name}')
        return existing_experiment
        
    except Exception as e:
        session.rollback()
        logging.error(f'Error updating experiment {experiment_name if "experiment_name" in locals() else "unknown"}: {e}')
        logging.error(f'Exception type: {type(e).__name__}')
        logging.error(f'Exception details: {str(e)}')
        import traceback
        logging.error(f'Traceback: {traceback.format_exc()}')
        raise e
    finally:
        session.close()

def remove_experiment_from_db(experiment_name: str):
    """Removes an existing experiment from the database."""
    from models_db import Experiment, CellObservation, Cell, Timepoint, RawImage, ProcessedImage
    from sqlalchemy.exc import IntegrityError
    
    session = SessionLocal()
    try:
        existing_experiment = session.query(Experiment).filter_by(name=experiment_name).first()
        if not existing_experiment:
            raise ValueError(f"Experiment with name '{experiment_name}' does not exist for removal")
        
        experiment_id = existing_experiment.id
        
        # Delete data in the correct order due to foreign key constraints
        # Start with partitioned table (cell_observations) which doesn't have CASCADE
        session.query(CellObservation).filter_by(experiment_id=experiment_id).delete()
        
        # Delete other related data (these should cascade, but we'll be explicit)
        session.query(RawImage).filter_by(experiment_id=experiment_id).delete()
        session.query(ProcessedImage).filter_by(experiment_id=experiment_id).delete()
        session.query(Cell).filter_by(experiment_id=experiment_id).delete()
        session.query(Timepoint).filter_by(experiment_id=experiment_id).delete()
        
        # Finally delete the experiment itself
        session.delete(existing_experiment)
        session.commit()
        
        logging.info(f"Successfully deleted experiment '{experiment_name}' from database")
        return True
        
    except IntegrityError as e:
        session.rollback()
        logging.error(f"Integrity error deleting experiment '{experiment_name}': {e}")
        raise ValueError(f"Could not delete experiment '{experiment_name}' due to database constraints: {e}")
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting experiment '{experiment_name}': {e}")
        raise ValueError(f"Could not delete experiment '{experiment_name}': {e}")
    finally:
        session.close()

def update_metadata(data_path: str):
    """Updates only the metadata of an existing experiment."""
    from models_db import Experiment
    
    session = SessionLocal()
    try:
        # Read metadata to identify experiment
        meta_name = [f for f in os.listdir(data_path) if 'metadata.json' in f.lower()]
        if len(meta_name) == 0:
            raise ValueError('No metadata.json file found in the provided data path')
        else:
            meta_name = meta_name[0]
        
        meta_path = os.path.join(data_path, meta_name)
        
        with open(meta_path, 'r') as file:
            metadata = json.load(file)
        
        experiment_name = metadata.get('experiment_name', 'unknown_experiment')
        existing_experiment = session.query(Experiment).filter_by(name=experiment_name).first()
        if not existing_experiment:
            raise ValueError(f"Experiment with name '{experiment_name}' does not exist for metadata update")
        
        # Update only metadata fields
        _update_experiment_metadata(existing_experiment, metadata, session)
        session.commit()
        logging.info(f'Successfully updated metadata for experiment: {experiment_name}')
        return existing_experiment
        
    except Exception as e:
        session.rollback()
        logging.error(f'Error updating metadata for experiment: {e}')
        raise e
    finally:
        session.close()