from typing import Annotated
from fastapi import FastAPI, HTTPException, Request, status,Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services.detector import checkEMail

app = FastAPI()

app.mount("/static",StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request : Request):
    return templates.TemplateResponse(request, "index.html")
  
  
@app.post("/check")
async def check(request: Request, emailText: str = Form(...), modelOption: str = Form(...)):
    prob = checkEMail(emailText,modelOption)
    
    result_text = "Phishing" if prob >= 0.5 else "Safe"
    
    # Convert probability to a percentage (0.46 -> 46)
    risk_score = int(round(float(prob) * 100))
    
    print(f"DEBUG: Sending to HTML -> Result: {result_text}, Score: {risk_score}")
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": result_text, 
        "risk_score": risk_score
    })