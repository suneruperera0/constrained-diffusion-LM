#!/usr/bin/env python3
"""
FastAPI backend for Constrained Diffusion LM React frontend.
Serves the React app and provides API endpoints for inference.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json

from constrained_diffusion_lm.data import Tokenizer
from constrained_diffusion_lm.data.constraints import lock_substring
from constrained_diffusion_lm.diffusion import get_schedule
from constrained_diffusion_lm.diffusion.sampler import ImprovedConstrainedSampler
from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.eval.edit_metrics import compute_edit_metrics
from constrained_diffusion_lm.utils.seed import get_device

# Global model components
MODEL = None
TOKENIZER = None
SCHEDULE = None
SAMPLER = None
DEVICE = None

# Request/Response models
class EditRequest(BaseModel):
    input_text: str
    locked_spans: str = ""
    diffusion_steps: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    edit_strength: float = 0.4

class EditResponse(BaseModel):
    generated_text: str
    visualization: str
    metrics: dict

def load_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Load the model and initialize components."""
    global MODEL, TOKENIZER, SCHEDULE, SAMPLER, DEVICE
    
    DEVICE = get_device("auto")
    print(f"Using device: {DEVICE}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Infer model config
    state_dict = checkpoint["model_state_dict"]
    dim = state_dict["token_embedding.weight"].shape[1]
    max_seq_len = state_dict["position_embedding.weight"].shape[0]
    num_layers = sum(1 for k in state_dict if "transformer.layers" in k and "self_attn.in_proj_weight" in k)
    num_heads = dim // 64 if dim >= 64 else dim // 32
    dim_feedforward = state_dict["transformer.layers.0.linear1.weight"].shape[0]
    
    # Load tokenizer
    TOKENIZER = Tokenizer("bert-base-uncased", max_length=max_seq_len)
    
    # Create model
    MODEL = TransformerDenoiser(
        vocab_size=TOKENIZER.vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        max_seq_len=max_seq_len,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    MODEL.load_state_dict(state_dict)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    # Create schedule and sampler
    SCHEDULE = get_schedule("cosine", 100)
    SAMPLER = ImprovedConstrainedSampler(
        model=MODEL,
        schedule=SCHEDULE,
        mask_token_id=TOKENIZER.mask_token_id,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    
    print(f"Model loaded: {MODEL.get_num_params():,} parameters")
    return True

def edit_text_api(request: EditRequest) -> EditResponse:
    """Edit text API endpoint logic."""
    global MODEL, TOKENIZER, SAMPLER, DEVICE
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.input_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    # Tokenize input
    token_ids = TOKENIZER.encode(
        request.input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
    )
    x_0 = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    
    # Create lock mask from locked spans
    lock_mask = torch.zeros_like(x_0, dtype=torch.bool)
    locked_texts = []
    
    if request.locked_spans.strip():
        for span in request.locked_spans.split(","):
            span = span.strip()
            if span and span.lower() in request.input_text.lower():
                span_lock_mask, _, found_spans = lock_substring(
                    request.input_text, span, TOKENIZER, seq_len=x_0.shape[1]
                )
                span_lock_tensor = span_lock_mask.unsqueeze(0).to(DEVICE)
                lock_mask = lock_mask | span_lock_tensor
                if found_spans:
                    locked_texts.append(span)
    
    # Update schedule
    global SCHEDULE
    SCHEDULE = get_schedule("cosine", request.diffusion_steps)
    SAMPLER.schedule = SCHEDULE
    SAMPLER.num_timesteps = request.diffusion_steps
    
    # Apply partial masking based on edit_strength
    edit_mask = ~lock_mask
    
    if request.edit_strength < 1.0:
        import random
        editable_positions = edit_mask[0].nonzero(as_tuple=True)[0].tolist()
        num_to_mask = max(1, int(len(editable_positions) * request.edit_strength))
        positions_to_mask = random.sample(editable_positions, num_to_mask)
        
        partial_edit_mask = torch.zeros_like(edit_mask)
        for pos in positions_to_mask:
            partial_edit_mask[0, pos] = True
        
        lock_mask = ~partial_edit_mask
    
    # Run editing
    with torch.no_grad():
        edited_ids = SAMPLER.edit(
            x_0=x_0,
            lock_mask=lock_mask,
            temperature=request.temperature,
            top_k=request.top_k if request.top_k > 0 else None,
            top_p=request.top_p if request.top_p < 1.0 else None,
            repetition_penalty=request.repetition_penalty,
        )
    
    # Decode output
    generated_text = TOKENIZER.decode(edited_ids[0].tolist())
    
    # Compute metrics
    metrics = compute_edit_metrics(
        original_ids=x_0,
        edited_ids=edited_ids,
        lock_mask=lock_mask,
    )
    
    # Create visualization
    hf_tokenizer = TOKENIZER.tokenizer
    original_tokens = hf_tokenizer.convert_ids_to_tokens(x_0[0].tolist())
    edited_tokens = hf_tokenizer.convert_ids_to_tokens(edited_ids[0].tolist())
    
    viz_parts = []
    lock_mask_list = lock_mask[0].tolist()
    
    for i, (orig, edit, locked) in enumerate(zip(
        original_tokens,
        edited_tokens,
        lock_mask_list
    )):
        if orig in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        if locked:
            viz_parts.append(
                f'<span style="background-color: #90EE90; padding: 2px 4px; margin: 1px; border-radius: 3px;">{edit}</span>'
            )
        elif orig != edit:
            viz_parts.append(
                f'<span style="background-color: #FFB6C1; padding: 2px 4px; margin: 1px; border-radius: 3px;">{edit}</span>'
            )
        else:
            viz_parts.append(
                f'<span style="background-color: #FFFACD; padding: 2px 4px; margin: 1px; border-radius: 3px;">{edit}</span>'
            )
    
    visualization = f"""
    <div style="font-family: monospace; line-height: 2; padding: 10px; background: #f5f5f5; border-radius: 8px;">
    {' '.join(viz_parts)}
    </div>
    
    <div style="margin-top: 10px; font-size: 12px;">
    <span style="background-color: #90EE90; padding: 2px 6px; border-radius: 3px;">🔒 Locked</span>
    <span style="background-color: #FFFACD; padding: 2px 6px; border-radius: 3px; margin-left: 8px;">📝 Unchanged</span>
    <span style="background-color: #FFB6C1; padding: 2px 6px; border-radius: 3px; margin-left: 8px;">✏️ Changed</span>
    </div>
    """
    
    return EditResponse(
        generated_text=generated_text,
        visualization=visualization,
        metrics={
            "constraint_fidelity": metrics.constraint_fidelity,
            "edit_rate": metrics.edit_rate,
            "num_locked_tokens": metrics.num_locked_tokens,
            "num_locked_preserved": metrics.num_locked_preserved,
        }
    )

# Create FastAPI app
app = FastAPI(title="Constrained Diffusion LM API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (React build)
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build)), name="static")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Serve React app."""
    index_path = frontend_build / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not built. Run 'npm run build' in frontend/ directory."}

@app.post("/api/edit", response_model=EditResponse)
async def edit_text_endpoint(request: EditRequest):
    """Edit text endpoint."""
    try:
        result = edit_text_api(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


