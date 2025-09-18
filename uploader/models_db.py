from sqlalchemy import Column, Integer, String, Numeric, Text, Boolean, DateTime, Interval, ForeignKey, BigInteger, Index
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP)
    cell_type = Column(String(255))
    condition = Column(String(255))
    condition_amount = Column(Numeric)
    condition_unit = Column(String(50))
    condition_time = Column(Interval)
    notes = Column(Text)
    meta = Column(JSONB)
    created_at = Column(TIMESTAMP, default=datetime.now())
    
    # Relationships
    timepoints = relationship("Timepoint", back_populates="experiment", cascade="all, delete-orphan")
    cells = relationship("Cell", back_populates="experiment", cascade="all, delete-orphan")
    raw_images = relationship("RawImage", back_populates="experiment", cascade="all, delete-orphan")
    processed_images = relationship("ProcessedImage", back_populates="experiment", cascade="all, delete-orphan")


class Timepoint(Base):
    __tablename__ = 'timepoints'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    timepoint_number = Column(Integer, nullable=False)
    timestamp_absolute = Column(TIMESTAMP)
    timestamp_relative = Column(Interval)  # time since experiment start
    
    # Relationships
    experiment = relationship("Experiment", back_populates="timepoints")
    cell_observations = relationship("CellObservation", back_populates="timepoint")
    raw_images = relationship("RawImage", back_populates="timepoint", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('idx_timepoints_exp_number', 'experiment_id', 'timepoint_number', unique=True),
    )


class Cell(Base):
    __tablename__ = 'cells'
    
    id = Column(BigInteger, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    cell_identifier = Column(String(255), nullable=False)  # your internal cell ID
    parent_cell_id = Column(BigInteger, ForeignKey('cells.id'))
    birth_timepoint_id = Column(Integer, ForeignKey('timepoints.id'))
    death_timepoint_id = Column(Integer, ForeignKey('timepoints.id'))
    generation = Column(Integer, nullable=False, default=0)
    created_at = Column(TIMESTAMP, default=datetime.now())

    # Relationships
    experiment = relationship("Experiment", back_populates="cells")
    parent_cell = relationship("Cell", remote_side=[id], backref="daughter_cells")
    birth_timepoint = relationship("Timepoint", foreign_keys=[birth_timepoint_id])
    death_timepoint = relationship("Timepoint", foreign_keys=[death_timepoint_id])
    observations = relationship("CellObservation", back_populates="cell")
    
    # Constraints
    __table_args__ = (
        Index('idx_cells_experiment', 'experiment_id'),
        Index('idx_cells_parent', 'parent_cell_id'),
        Index('idx_cells_lineage', 'experiment_id', 'generation'),
        Index('idx_cells_lifetime', 'birth_timepoint_id', 'death_timepoint_id'),
        Index('idx_cells_exp_identifier', 'experiment_id', 'cell_identifier', unique=True),
    )


class ObservableType(Base):
    __tablename__ = 'observable_types'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    unit = Column(String(50))
    data_type = Column(String(20), nullable=False, default='numeric')  # numeric, text, boolean
    description = Column(Text)
    
    # Relationships
    observations = relationship("CellObservation", back_populates="observable_type")


class CellObservation(Base):
    __tablename__ = 'cell_observations'
    
    id = Column(BigInteger, primary_key=True)
    experiment_id = Column(Integer, nullable=False)
    timepoint_id = Column(Integer, ForeignKey('timepoints.id'), nullable=False)
    cell_id = Column(BigInteger, ForeignKey('cells.id'), nullable=False)
    observable_type_id = Column(Integer, ForeignKey('observable_types.id'), nullable=False)
    value_numeric = Column(Numeric)
    value_text = Column(Text)
    value_boolean = Column(Boolean)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    timepoint = relationship("Timepoint", back_populates="cell_observations")
    cell = relationship("Cell", back_populates="observations")
    observable_type = relationship("ObservableType", back_populates="observations")
    
    # Constraints
    __table_args__ = (
        Index('idx_obs_cell_time', 'cell_id', 'timepoint_id'),
        Index('idx_obs_exp_time_obs', 'experiment_id', 'timepoint_id', 'observable_type_id'),
        Index('idx_obs_cell_observable', 'cell_id', 'observable_type_id'),
        Index('idx_obs_timepoint', 'timepoint_id'),
        {'postgresql_partition_by': 'HASH (experiment_id)'}
    )


class RawImage(Base):
    __tablename__ = 'raw_images'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    timepoint_id = Column(Integer, ForeignKey('timepoints.id', ondelete='CASCADE'), nullable=False)
    file_path = Column(Text, nullable=False)  # path to image file
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="raw_images")
    timepoint = relationship("Timepoint", back_populates="raw_images")
    
    # Constraints
    __table_args__ = (
        Index('idx_raw_images_exp_time_path', 'experiment_id', 'timepoint_id', 'file_path', unique=True),
    )


class ProcessedImage(Base):
    __tablename__ = 'processed_images'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    file_path = Column(Text, nullable=False)  # path to processed image file
    image_type = Column(String(100))  # "AC", "DC" or "Segmentation"
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="processed_images")
    
    # Constraints
    __table_args__ = (
        Index('idx_processed_images_exp_path', 'experiment_id', 'file_path', unique=True),
    )


# Data classes for working with parsed data from functions
class CellData:
    """Data structure for cell data parsed from read_cell_data function"""
    def __init__(self, cell_dict):
        self.name = cell_dict.get('name', '')
        self.parent = cell_dict.get('parent', None)
        self.position_x = cell_dict.get('X', [])
        self.position_y = cell_dict.get('Y', [])
        
        # Store all other measurements as a dictionary
        self.measurements = {}
        for key, value in cell_dict.items():
            if key not in ['name', 'parent', 'X', 'Y']:
                self.measurements[key] = value
    
    @property
    def cell_identifier(self):
        return self.name
    
    @property
    def parent_identifier(self):
        return self.parent
    
    def get_timepoint_measurements(self, timepoint_index):
        """Get measurements for a specific timepoint"""
        measurements = {}
        for key, values in self.measurements.items():
            if hasattr(values, '__len__') and len(values) > timepoint_index:
                measurements[key] = values[timepoint_index]
        
        # Add position data
        if len(self.position_x) > timepoint_index:
            measurements['position_x'] = self.position_x[timepoint_index]
        if len(self.position_y) > timepoint_index:
            measurements['position_y'] = self.position_y[timepoint_index]
            
        return measurements


class TimepointData:
    """Data structure for timepoint data parsed from read_timepoints function"""
    def __init__(self, times, intervals):
        self.times = times  # absolute timestamps
        self.intervals = intervals  # relative intervals from start
    
    def get_timepoint_pairs(self):
        """Returns list of (timepoint_number, absolute_time, relative_interval) tuples"""
        pairs = []
        for i, (time, interval) in enumerate(zip(self.times, self.intervals)):
            pairs.append((i, time, interval))
        return pairs


class ImagePathData:
    """Data structure for image paths parsed from read_image_paths function"""
    def __init__(self, ac_path=None, dc_path=None, seg_path=None, raws_paths=None):
        self.ac_path = ac_path
        self.dc_path = dc_path
        self.seg_path = seg_path
        self.raws_paths = raws_paths or []
    
    def get_processed_images(self):
        """Returns list of (image_type, file_path) tuples for processed images"""
        processed = []
        if self.ac_path:
            processed.append(('AC', self.ac_path))
        if self.dc_path:
            processed.append(('DC', self.dc_path))
        if self.seg_path:
            processed.append(('Segmentation', self.seg_path))
        return processed
    
    def get_raw_images(self):
        """Returns list of raw image file paths"""
        return self.raws_paths if self.raws_paths else []
