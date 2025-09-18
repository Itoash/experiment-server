import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import json
from typing import List
from datetime import datetime, timedelta

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://researcher:biodinamica2025!@db:5432/experiment_db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

def load_experiment(data_path:str):
    # Placeholder for actual loading logic
    print(f"Loading experiment data from {data_path}")
    

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
                    print(f'Error at attribue {name}')
            celldict.update({name:np.array(data)})
        celldict.update({'name':lines[idx].strip()})
        if len(celldict['name'].split('_'))>2:
            # Means cell is a daughter cell
            celldict.update({'parent': '_'.join(celldict['name'].split('_')[:-1])})
        else:
            celldict.update({'parent': None})
        cells.append(celldict)
    return cells

def read_timepoints(path):
    """Reads timepoint data from a text file and returns a list of dictionaries, each representing a timepoint.
    timepoints: List of timestamps in date-time format.
    intervals: List of time intervals in minutes since start of experiment.
    """
    metadata_path = [f for f in os.listdir(path) if 'metadata.json' in f.lower()][0]
    with open(os.path.join(path, metadata_path)) as file:
        metadata = json.load(file)
    exp_name = metadata.get('name', 'unknown_experiment')

    csv_path = [f for f in os.listdir(path) if exp_name in f and f.endswith('.csv')][0]
    with open(os.path.join(path, csv_path),'r') as file:
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
    print(times,intervals)
    return times, intervals
def read_image_paths(path):
    """Reads image paths from metadata and returns structured data about images."""
    metadata_path = [f for f in os.listdir(path) if 'metadata.json' in f.lower()][0]
    with open(os.path.join(path, metadata_path)) as file:
        metadata = json.load(file)
    exp_name = metadata.get('experiment_name', 'unknown_experiment')
    
    # Look for processed images in the expected locations
    processed_images_base = os.path.join(os.path.dirname(path), '..', 'images', 'processed', exp_name)
    raw_images_base = os.path.join(os.path.dirname(path), '..', 'images', 'raws')
    
    # Check for processed images
    AC_path = None
    DC_path = None
    seg_path = None
    
    if os.path.exists(processed_images_base):
        for f in os.listdir(processed_images_base):
            if f.lower().endswith('.tif'):
                if 'ac' in f.lower():
                    AC_path = os.path.join(processed_images_base, f)
                elif 'dc' in f.lower():
                    DC_path = os.path.join(processed_images_base, f)
                elif 'seg' in f.lower():
                    seg_path = os.path.join(processed_images_base, f)
    
    # Check for raw image directories
    raws_paths = []
    if os.path.exists(raw_images_base):
        for item in os.listdir(raw_images_base):
            item_path = os.path.join(raw_images_base, item)
            if os.path.isdir(item_path):
                raws_paths.append(item_path)
    
    return {
        'AC': AC_path,
        'DC': DC_path, 
        'segmentation': seg_path,
        'raw': raws_paths,
        'metadata': metadata
    }

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
                        time_part = parts[i + 1]
                    # AM/PM should be after that or separate
                    if i + 2 < len(parts):
                        am_pm = parts[i + 2].strip()
                    elif ' ' in dir_name:
                        # AM/PM might be separated by space
                        am_pm_split = dir_name.split(' ')
                        if len(am_pm_split) > 1:
                            am_pm = am_pm_split[-1].strip()
                    break
            
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
                        print(f"Could not parse time format: {time_part}")
                        continue
                    
                    # Create datetime string
                    datetime_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                    if am_pm and am_pm.upper() in ['AM', 'PM']:
                        datetime_str += f" {am_pm.upper()}"
                        dt = datetime.strptime(datetime_str, '%Y-%m-%d %I:%M:%S %p')
                    else:
                        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                    
                    timestamp = dt.timestamp()
                    
                    # Find the closest timepoint
                    if timepoints:
                        time_diffs = [abs(tp - timestamp) for tp in timepoints]
                        closest_index = time_diffs.index(min(time_diffs))
                        timepoint_indices.append(closest_index)
                    else:
                        timepoint_indices.append(0)  # Default to first timepoint
                        
                except (ValueError, IndexError) as e:
                    print(f"Could not parse date/time from directory name: {dir_name}, error: {e}")
                    timepoint_indices.append(0)  # Default to first timepoint
            else:
                print(f"Could not find date/time parts in directory name: {dir_name}")
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
    from models import Experiment
    from sqlalchemy.exc import IntegrityError
    
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    
    # Check if experiment already exists
    experiment_name = metadata.get('name', 'unknown_experiment')
    existing_experiment = session.query(Experiment).filter_by(name=experiment_name).first()
    if existing_experiment:
        raise ValueError(f"Experiment with name '{experiment_name}' already exists")
    
    # Parse dates
    start_time = None
    end_time = None
    
    if 'start_time' in metadata:
        start_time = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00'))
    elif 'date' in metadata:
        start_time = datetime.fromisoformat(metadata['date'].replace('Z', '+00:00'))
    
    if 'end_time' in metadata:
        end_time = datetime.fromisoformat(metadata['end_time'].replace('Z', '+00:00'))
    
    experiment = Experiment(
        name=metadata.get('name', 'unknown_experiment'),
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


def create_timepoints_from_data(experiment_id: int, times: List[float], intervals: List[float]):
    """Creates Timepoint objects from timepoint data."""
    from models import Timepoint
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


def create_cells_from_data(experiment_id: int, cells_data: List[dict]):
    """Creates Cell objects from cell data."""
    from models import Cell
    
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
    from models import CellObservation
    
    observations = []
    
    for cell_dict in cells_data:
        cell_name = cell_dict.get('name', '')
        
        # Process each measurement type
        for measurement_name, values in cell_dict.items():
            if measurement_name in ['name', 'parent', 'X', 'Y']:
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
                    if i < len(timepoints):
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
    from models import ProcessedImage
    
    processed_images = []
    
    image_types = {
        'AC': image_data.get('AC'),
        'DC': image_data.get('DC'),
        'Segmentation': image_data.get('segmentation')
    }
    
    for image_type, file_path in image_types.items():
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
    from models import RawImage
    
    raw_images = []
    
    for raw_path, timepoint_idx in zip(raw_paths, timepoint_indices):
        if timepoint_idx < len(timepoints):
            timepoint_id = timepoints[timepoint_idx].id
            
            if os.path.isdir(raw_path):
                # Add all image files in the directory
                for filename in os.listdir(raw_path):
                    if filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(raw_path, filename)
                        raw_image = RawImage(
                            experiment_id=experiment_id,
                            timepoint_id=timepoint_id,
                            file_path=file_path
                        )
                        raw_images.append(raw_image)
            elif os.path.isfile(raw_path):
                raw_image = RawImage(
                    experiment_id=experiment_id,
                    timepoint_id=timepoint_id,
                    file_path=raw_path
                )
                raw_images.append(raw_image)
    
    return raw_images

def set_parent_cell_ids(cells, cell_name_to_id):
    """Sets parent_cell_id for each Cell object based on cell_name_to_id mapping."""
    for cell in cells:
        if cell.parent_cell_id is None and cell.cell_identifier in cell_name_to_id:
            parent_name = cell.cell_identifier.rsplit('_', 1)[0]  # Assuming parent name is prefix before last '_'
            if parent_name in cell_name_to_id:
                cell.parent_cell_id = cell_name_to_id[parent_name]
    return cells

def set_birth_death_times(cells, cells_data, timepoints):
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
    
def load_experiment_complete(data_path: str):
    """Complete pipeline to load experiment data and create all SQLAlchemy objects.
       This is only used upon experiment creation, so watcher should call this when detecting metadata
       with the 'create' flag in 'last_operation'
    """
    from models import ObservableType
    from sqlalchemy.exc import IntegrityError
    
    # Read all data
    metadata_file = os.path.join(data_path, [f for f in os.listdir(data_path) if 'metadata.json' in f.lower()][0])
    cells_data = read_cell_data(os.path.join(data_path, [f for f in os.listdir(data_path) if f.endswith('.csv')][0]))
    times, intervals = read_timepoints(data_path)
    image_data = read_image_paths(data_path)
    
    try:
        # Create experiment
        experiment = create_experiment_from_metadata(metadata_file)
        session.add(experiment)
        session.flush()  # Get the experiment ID
    except IntegrityError:
        session.rollback()
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            experiment_name = metadata.get('name', 'unknown_experiment')
            raise ValueError(f"Cannot load experiment: an experiment with name '{experiment_name}' already exists")
    except ValueError as e:
        session.rollback()
        raise e
    
    # Create timepoints
    timepoints = create_timepoints_from_data(experiment.id, times, intervals)
    session.add_all(timepoints)
    session.flush()  # Get timepoint IDs
    
    # Create cells and get their IDs
    cells, cell_name_to_id = create_cells_from_data(experiment.id, cells_data)
    db_cells = session.query(cells[0].__class__).filter_by(experiment_id=experiment.id).all()
    print(f'Number of cells in DB: {len(db_cells)}')
    cells = set_parent_cell_ids(cells, cell_name_to_id)
    cells = set_birth_death_times(cells, cells_data, timepoints)
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
    
    return experiment

def update_existing_experiment(data_path: str):
    """Updates an existing experiment with new data.
       This is used when the watcher detects metadata with the 'update' flag in 'last_operation'
    """
    from models import Experiment
    from sqlalchemy.exc import IntegrityError
    
    # Read metadata to identify experiment
    metadata_file = os.path.join(data_path, [f for f in os.listdir(data_path) if 'metadata.json' in f.lower()][0])
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
    
    experiment_name = metadata.get('name', 'unknown_experiment')
    existing_experiment = session.query(Experiment).filter_by(name=experiment_name).first()
    if not existing_experiment:
        raise ValueError(f"Experiment with name '{experiment_name}' does not exist for update")
    
    # For simplicity, we will just update the notes and end_time if provided
    if 'notes' in metadata:
        existing_experiment.notes = metadata['notes']
    if 'end_time' in metadata:
        existing_experiment.end_time = datetime.fromisoformat(metadata['end_time'].replace('Z', '+00:00'))
    
    session.add(existing_experiment)
    session.commit()
    
    return existing_experiment


def update_metadata(data_path: str):
    """Updates only the metadata of an existing experiment.
    This is used when the watcher detects metadata with the 'metadata_update' flag in 'last_operation'
    """
    return update_existing_experiment(data_path)