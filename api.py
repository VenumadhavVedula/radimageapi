from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
import os

from dcmtopng import dicom2jpeg

app = FastAPI(
    title="Chest X-Ray Analysis API",
    description="API for analyzing chest X-ray images using torchxrayvision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/analyzedcm/", tags=["X-Ray Analysis"])
async def analyze_xray_image(file: UploadFile = File(...)):
    # Import heavy dependencies only when this endpoint is called
    import tempfile
    from analyzexray import analyze_xray
    from heatmapxray import heatmap_xray

    try:
        # Create temporary files for input and heatmap
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_dcm_file, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as heatmap_temp:
            
            # Handle input file
            contents = await file.read()
            temp_dcm_file.write(contents)
            dicom2jpeg(temp_dcm_file.name, temp_file.name)
            temp_path = temp_file.name
            heatmap_path = heatmap_temp.name

        # Get analysis results
        results = analyze_xray(temp_path)
        
        # Generate heatmap with specific output path
        heatmap_xray(temp_path, heatmap_path)
        
        # Read the generated heatmap and convert to base64
        with open(heatmap_path, "rb") as heatmap_file:
            heatmap_base64 = base64.b64encode(heatmap_file.read()).decode('utf-8')
        
        # Clean up temporary files
        os.unlink(temp_path)
        os.unlink(heatmap_path)
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "predictions": {k: float(v) for k, v in results.items()},
                "heatmap": heatmap_base64
            }
        )
    except Exception as e:
        # Clean up temporary files in case of error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        if 'heatmap_path' in locals():
            os.unlink(heatmap_path)
            
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/analyze/", tags=["X-Ray Analysis"])
async def analyze_xray_image(file: UploadFile = File(...)):
    # Import heavy dependencies only when this endpoint is called
    import tempfile
    from analyzexray import analyze_xray
    from heatmapxray import heatmap_xray

    try:
        # Create temporary files for input and heatmap
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as heatmap_temp:
            
            # Handle input file
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
            heatmap_path = heatmap_temp.name

        # Get analysis results
        results = analyze_xray(temp_path)
        
        # Generate heatmap with specific output path
        heatmap_xray(temp_path, heatmap_path)
        
        # Read the generated heatmap and convert to base64
        with open(heatmap_path, "rb") as heatmap_file:
            heatmap_base64 = base64.b64encode(heatmap_file.read()).decode('utf-8')
        
        # Clean up temporary files
        os.unlink(temp_path)
        os.unlink(heatmap_path)
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "predictions": {k: float(v) for k, v in results.items()},
                "heatmap": heatmap_base64
            }
        )
    except Exception as e:
        # Clean up temporary files in case of error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        if 'heatmap_path' in locals():
            os.unlink(heatmap_path)
            
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/predictdcm/", tags=["X-Ray Analysis Predictions"])
async def analyze_xray_image(file: UploadFile = File(...)):
    # Import heavy dependencies only when this endpoint is called
    import tempfile
    from analyzexray import analyze_xray

    try:
        # Create temporary files for input and heatmap
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            
            # Handle input file
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Get analysis results
        results = analyze_xray(temp_path)
        
        # Clean up temporary files
        os.unlink(temp_path)
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "predictions": {k: float(v) for k, v in results.items()},
            }
        )
    except Exception as e:
        # Clean up temporary files in case of error
        if 'temp_path' in locals():
            os.unlink(temp_path)
            
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "service is running"}