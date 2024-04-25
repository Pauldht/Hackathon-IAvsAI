from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://user-amy12-681443-user.user.lab.sspcloud.fr",
    "https://user-amy12-681443-0.user.lab.sspcloud.fr",
    "https://user-amy12-681443-0.user.lab.sspcloud.fr/proxy/8000/post_data_searchbar",
    "https://user-amy12-681443-user.user.lab.sspcloud.fr/", 
    "http://localhost:3000",
    "http://localhost:8000/",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app = FastAPI(openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],  # Séparer chaque méthode par une virgule
    allow_headers=["*"]
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/get_data_searchbar")
async def get_data_searchbar():
    # Logique de récupération des données
    return {"message": "GET request received"}


@app.post("/get_data_searchbar")
async def post_data_searchbar(texte: dict):
    # Logique de traitement des données envoyées
    print(texte)
    return texte["texte"]
