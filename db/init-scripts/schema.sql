-- Cell Tracking Database Schema for Time Series Experiments

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    cell_type VARCHAR(255),
    condition VARCHAR(255),
    condition_amount NUMERIC,
    condition_unit VARCHAR(50),
    condition_time INTERVAL,
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Timepoints for each experiment
CREATE TABLE IF NOT EXISTS timepoints (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    timepoint_number INTEGER NOT NULL,
    timestamp_absolute TIMESTAMPTZ,
    timestamp_relative INTERVAL, -- time since experiment start
    UNIQUE(experiment_id, timepoint_number)
);

-- Cells table with lifetime tracking
CREATE TABLE IF NOT EXISTS cells (
    id BIGSERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    cell_identifier VARCHAR(255) NOT NULL, -- your internal cell ID
    parent_cell_id BIGINT REFERENCES cells(id),
    birth_timepoint_id INTEGER REFERENCES timepoints(id),
    death_timepoint_id INTEGER REFERENCES timepoints(id),
    generation INTEGER NOT NULL DEFAULT 0, -- calculated from parent relationships
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(experiment_id, cell_identifier)
);

-- Observable types (measurements you're taking)
CREATE TABLE IF NOT EXISTS observable_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    unit VARCHAR(50),
    data_type VARCHAR(20) NOT NULL DEFAULT 'numeric', -- numeric, text, boolean
    description TEXT
);

-- Main time series data table - partitioned by experiment and timepoint
CREATE TABLE IF NOT EXISTS cell_observations (
    id BIGSERIAL,
    experiment_id INTEGER NOT NULL,
    timepoint_id INTEGER NOT NULL REFERENCES timepoints(id),
    cell_id BIGINT NOT NULL REFERENCES cells(id),
    observable_type_id INTEGER NOT NULL REFERENCES observable_types(id),
    value_numeric NUMERIC,
    value_text TEXT,
    value_boolean BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, experiment_id, cell_id)
) PARTITION BY HASH (experiment_id);


--- Table for raw image folder paths (points to a directory)
CREATE TABLE IF NOT EXISTS raw_images (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    timepoint_id INTEGER NOT NULL REFERENCES timepoints(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL, -- path to image file
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(experiment_id, timepoint_id, file_path)
);
--- Table for processed image paths (points to a multi-tiff file)
CREATE TABLE IF NOT EXISTS processed_images (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL, -- path to processed image file
    image_type VARCHAR(100), -- "AC", "DC" or "Segmentation"
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(experiment_id, file_path)
);

-- Create initial partitions (you can add more as experiments grow)
CREATE TABLE IF NOT EXISTS cell_observations_p0 PARTITION OF cell_observations
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE IF NOT EXISTS cell_observations_p1 PARTITION OF cell_observations
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE IF NOT EXISTS cell_observations_p2 PARTITION OF cell_observations
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE IF NOT EXISTS cell_observations_p3 PARTITION OF cell_observations
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_experiments_start_time ON experiments(start_time);
CREATE INDEX IF NOT EXISTS idx_timepoints_exp_number ON timepoints(experiment_id, timepoint_number);
CREATE INDEX IF NOT EXISTS idx_cells_experiment ON cells(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cells_parent ON cells(parent_cell_id) WHERE parent_cell_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cells_lineage ON cells(experiment_id, generation);
CREATE INDEX IF NOT EXISTS idx_cells_lifetime ON cells(birth_timepoint_id, death_timepoint_id);

-- Indexes on partitioned table
CREATE INDEX IF NOT EXISTS idx_obs_cell_time ON cell_observations(cell_id, timepoint_id);
CREATE INDEX IF NOT EXISTS idx_obs_exp_time_obs ON cell_observations(experiment_id, timepoint_id, observable_type_id);
CREATE INDEX IF NOT EXISTS idx_obs_cell_observable ON cell_observations(cell_id, observable_type_id);
CREATE INDEX IF NOT EXISTS idx_obs_timepoint ON cell_observations(timepoint_id);

-- Useful views for common queries
CREATE VIEW cell_lineages AS
WITH RECURSIVE lineage AS (
    -- Base case: root cells (no parent)
    SELECT id, experiment_id, cell_identifier, parent_cell_id, 
           ARRAY[id] as lineage_path, 0 as depth
    FROM cells 
    WHERE parent_cell_id IS NULL
    
    UNION ALL
    
    -- Recursive case: cells with parents
    SELECT c.id, c.experiment_id, c.cell_identifier, c.parent_cell_id,
           l.lineage_path || c.id, l.depth + 1
    FROM cells c
    JOIN lineage l ON c.parent_cell_id = l.id
)
SELECT * FROM lineage;

-- View for active cells at each timepoint
CREATE VIEW active_cells_by_timepoint AS
SELECT 
    c.experiment_id,
    t.id as timepoint_id,
    t.timepoint_number,
    c.id as cell_id,
    c.cell_identifier,
    c.generation
FROM cells c
CROSS JOIN timepoints t
WHERE c.experiment_id = t.experiment_id
AND (c.birth_timepoint_id IS NULL OR c.birth_timepoint_id <= t.id)
AND (c.death_timepoint_id IS NULL OR c.death_timepoint_id > t.id);

-- Functions for data integrity
CREATE OR REPLACE FUNCTION update_cell_generation()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.parent_cell_id IS NOT NULL THEN
        SELECT generation + 1 INTO NEW.generation
        FROM cells 
        WHERE id = NEW.parent_cell_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_generation_trigger
    BEFORE INSERT OR UPDATE ON cells
    FOR EACH ROW
    EXECUTE FUNCTION update_cell_generation();

-- Insert some sample observable types
-- INSERT INTO observable_types (name, unit, data_type, description) VALUES
-- ('cell_area', 'μm²', 'numeric', 'Cell area in square micrometers'),
-- ('cell_volume', 'μm³', 'numeric', 'Cell volume in cubic micrometers'),
-- ('fluorescence_intensity', 'AU', 'numeric', 'Fluorescence intensity in arbitrary units'),
-- ('cell_velocity', 'μm/min', 'numeric', 'Cell movement velocity'),
-- ('cell_shape_factor', '', 'numeric', 'Circularity/shape factor (0-1)'),
-- ('mitotic_phase', '', 'text', 'Current phase of mitosis'),
-- ('is_dividing', '', 'boolean', 'Whether cell is currently dividing'),
-- ('protein_concentration', 'nM', 'numeric', 'Concentration of tracked protein');

INSERT INTO observable_types (name,unit,data_type,description) VALUES
('position_x', 'pixels', 'numeric', 'Position of the cell in the X axis'),
('position_y', 'pixels', 'numeric', 'Position of the cell in the Y axis'),
('area', 'pixels', 'numeric', 'Area of the cell'),
('Ellipse angle', 'degrees', 'numeric', 'Angle of the fitted ellipse'),
('Ellipse major', 'pixels', 'numeric', 'Length of the major axis of the fitted ellipse'),
('Ellipse minor', 'pixels', 'numeric', 'Length of the minor axis of the fitted ellipse'),
('DC mean', 'intensity', 'numeric', 'Mean intensity of the averaged image (DC)'),
('DC min', 'intensity', 'numeric', 'Minimum intensity of the averaged image (DC)'),
('DC max', 'intensity', 'numeric', 'Maximum intensity of the averaged image (DC)'),
('AC mean', 'amplitude', 'numeric', 'Mean amplitude of the amplitude image (AC)'),
('AC min', 'amplitude', 'numeric', 'Minimum amplitude of the amplitude image (AC)'),
('AC max', 'amplitude', 'numeric', 'Maximum amplitude of the amplitude image (AC)'),
('AC interior area', 'pixels', 'numeric', 'Area of the interior region of the amplitude image (AC)'),
('AC interior mean', 'amplitude', 'numeric', 'Mean amplitude of the interior region of the amplitude image (AC)'),
('AC contour mean', 'amplitude', 'numeric', 'Mean amplitude of the contour region of the amplitude image (AC)'),
('AC interior/back mean ratio', 'ratio (amplitude/amplitude)', 'numeric', 'Ratio of the mean amplitude of the interior to the back region (AC)'),
('AC interior/back contrast', 'ratio (amplitude/amplitude)', 'numeric', 'Contrast of the interior and back regions (AC)'),
('AC interior/contour contrast', 'ratio (amplitude/amplitude)', 'numeric', 'Contrast of the interior and contour regions (AC)'),
('AC?DC mean ratio', 'ratio (amplitude/intensity)', 'numeric', 'Ratio of the mean amplitude (AC) to the mean intensity (DC)'),
('AC solidity', 'ratio (pixels/pixels)', 'numeric', 'Solidity of the amplitude image (interior area/total area) (AC)');