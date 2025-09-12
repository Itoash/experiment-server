# PROJECT OUTLINE

## App

- Serves the interactive webpage via PYthon/something else
- Talks to the database for measurements
- Mounts the `images` directory for image visualisation
- Has no write capabilities to either the database or the file system

## Uploader

- Serves a webpage for uploading image data/experiment data
- Should have a form-like structure for different required fields
- Images are optional
- Uses `rsync` asynchronously to pull data from the host computer, puts it in a temp `uploads` directory
- Also adds upload metadata (TBD)

## Dataloader

- Sole writer to database
- Watches the `/uploads` folder for updates
- Ingests full files (separated image data+ CSV/JSON files per experiment)
- Moves them to full storage and completes the task (to `/images` folder)
- Adds an ok or error message to the `/completed` directory

## Database

- Backbone of server
- Gets updates from *dataloader*
- Is queried by *App*. for data to visualise and image paths to retrieve (read)
- Is queried by *Uploader* to check for experiment existence beforehand (read)
- Is queried by *Dataloader* for loading data and daa existence(read/write)
