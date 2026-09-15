import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from main import _load_dotenv
_load_dotenv()

from readers import load_user_history, load_evidence_requirements
from history import get_history_risk_flags
from validator import validate
from vision import create_genai_client, run_vision_pass
from output import format_row

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's local we can allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load global resources
history_df = load_user_history()
requirements_df = load_evidence_requirements()
client = create_genai_client()

@app.post("/verify-claim")
async def verify_claim(
    image: UploadFile = File(...),
    user_claim: str = Form(...),
    claim_object: str = Form(...),
):
    print(f"!!! RECEIVING REQUEST: {claim_object} !!!", flush=True)
    # save uploaded image to a temp file
    ext = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await image.read())
        temp_image_path = tmp.name
        
    try:
        user_id = "user_demo"
        
        print("!!! STARTING VISION PASS !!!", flush=True)
        
        # Run vision pass (synchronous, but in a real app might be async)
        # Note: we should run synchronous blocking operations in a thread pool ideally,
        # but for this demo this is fine.
        extraction = await asyncio.to_thread(
            run_vision_pass,
            temp_image_path,
            user_claim,
            user_id,
            claim_object,
            requirements_df,
            client,
        )
        
        if extraction.get("is_malicious_prompt"):
            raise HTTPException(status_code=400, detail="MALICIOUS_PROMPT_DETECTED")
        
        history_flags = get_history_risk_flags(user_id, history_df)
        
        validated_result, _fired_rules = validate(
            extraction,
            history_flags,
            claim_object=claim_object,
            requirements_df=requirements_df,
            num_images=1,
        )
        
        claim_row = {
            "user_id": user_id,
            "image_paths": temp_image_path,
            "user_claim": user_claim,
            "claim_object": claim_object
        }
        
        output_row = format_row(
            claim_row,
            extraction,
            history_flags,
            validated_result,
        )
        
        return output_row
        
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
