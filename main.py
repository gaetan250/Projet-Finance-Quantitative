import os
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from routes import data_processing, models_garch, models_lstm, sharpe, sharpe_cvar

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    """Vérifie la présence des fichiers critiques du serveur."""
    
    routes_folder = "routes"
    route_files = ["data_processing.py", "models_garch.py", "models_lstm.py", "sharpe.py", "sharpe_cvar.py"]

    missing_routes = [file for file in route_files if not os.path.exists(os.path.join(routes_folder, file))]

    if missing_files or missing_routes:
        return {
            "status": "error",
            "missing_files": missing_files,
            "missing_routes": missing_routes
        }

    return {"status": "ok"}

@app.get("/readme")
def read_readme(request: Request):
    """Affiche le contenu du README.md."""
    readme_path = "README.md"
    
    if not os.path.exists(readme_path):
        raise HTTPException(status_code=404, detail="README.md non trouvé")
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return templates.TemplateResponse("readme.html", {"request": request, "content": content})

# Inclusion des routes existantes
app.include_router(data_processing.router, prefix="/fetch-data")
app.include_router(models_garch.router, prefix="/garch")
app.include_router(models_lstm.router, prefix="/lstm-garch-cvi")
app.include_router(sharpe.router, prefix="/sharpe")
app.include_router(sharpe_cvar.router, prefix="/sharpe-cvar")
