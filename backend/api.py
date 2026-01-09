"""
FastAPI backend for Constrained Diffusion LM.

Uses BertDiffusionLM - leverages pretrained BERT MLM with timestep conditioning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from transformers import BertTokenizer
from constrained_diffusion_lm.models.bert_diffusion import BertDiffusionLM
from constrained_diffusion_lm.diffusion import get_schedule


# Global state
model: Optional[BertDiffusionLM] = None
tokenizer: Optional[BertTokenizer] = None
schedule = None
device = None


class EditRequest(BaseModel):
    text: str
    locked_spans: List[str] = []
    temperature: float = 0.8
    num_steps: int = 20


class EditResponse(BaseModel):
    output: str
    locked_preserved: bool
    tokens_changed: int
    tokens_total: int


def load_model(checkpoint_path: Optional[str] = None):
    """Load BertDiffusionLM model."""
    global model, tokenizer, schedule, device
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    schedule = get_schedule("cosine", 100)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Get max_timesteps from checkpoint config
        config = checkpoint.get("config", {})
        max_timesteps = config.get("max_timesteps", 1000)
        model = BertDiffusionLM(freeze_bert=True, max_timesteps=max_timesteps)
        # Checkpoint only contains trainable parts (timestep_embedding, time_norm)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            # Load only the trainable components
            model.timestep_embedding.load_state_dict(checkpoint["timestep_embedding"])
            model.time_norm.load_state_dict(checkpoint["time_norm"])
    else:
        print("Using fresh BertDiffusionLM (pretrained BERT + untrained timestep embedding)")
        model = BertDiffusionLM(freeze_bert=True)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")


def edit_text(
    text: str, 
    locked_spans: List[str], 
    temperature: float = 0.8,
    num_steps: int = 20,
) -> dict:
    """
    Edit text using diffusion while preserving locked spans.
    
    Process:
    1. Tokenize input text
    2. Identify locked token positions
    3. Mask all non-locked tokens
    4. Iteratively unmask using diffusion (confidence-based)
    5. Return final text
    """
    global model, tokenizer, schedule, device
    
    # Tokenize
    encoded = tokenizer.encode(text, add_special_tokens=True)
    x = torch.tensor([encoded], device=device)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    original_tokens = tokens.copy()
    seq_len = len(tokens)
    
    # Create lock mask (True = locked, don't change)
    lock_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    lock_mask[0] = True   # [CLS]
    lock_mask[-1] = True  # [SEP]
    
    # Find and lock specified spans
    for span in locked_spans:
        if not span.strip():
            continue
        span_tokens = tokenizer.tokenize(span)
        for i in range(seq_len - len(span_tokens) + 1):
            if tokens[i:i+len(span_tokens)] == span_tokens:
                for j in range(len(span_tokens)):
                    lock_mask[i+j] = True
    
    # Get editable positions
    editable_mask = ~lock_mask
    editable_positions = editable_mask.nonzero().squeeze(-1).tolist()
    if isinstance(editable_positions, int):
        editable_positions = [editable_positions]
    
    if len(editable_positions) == 0:
        # Nothing to edit
        return {
            "output": text,
            "locked_preserved": True,
            "tokens_changed": 0,
            "tokens_total": 0,
        }
    
    # Start with all editable tokens masked
    x_t = x.clone()
    mask_token_id = tokenizer.mask_token_id
    for pos in editable_positions:
        x_t[0, pos] = mask_token_id
    
    # Diffusion sampling: iteratively unmask tokens
    # Use confidence-based unmasking (unmask highest confidence first)
    num_to_unmask = len(editable_positions)
    tokens_per_step = max(1, num_to_unmask // num_steps)
    
    still_masked = set(editable_positions)
    
    with torch.no_grad():
        for step in range(num_steps):
            if not still_masked:
                break
            
            # Get predictions (no timestep needed - BERT MLM handles it directly)
            logits = model(x_t, attention_mask=torch.ones_like(x_t))
            
            # Compute confidence for each masked position
            confidences = []
            for pos in still_masked:
                probs = torch.softmax(logits[0, pos] / temperature, dim=-1)
                max_prob = probs.max().item()
                confidences.append((pos, max_prob, probs))
            
            # Sort by confidence (highest first)
            confidences.sort(key=lambda x: x[1], reverse=True)
            
            # Unmask top-k most confident tokens this step
            num_unmask_now = min(tokens_per_step, len(still_masked))
            if step == num_steps - 1:
                num_unmask_now = len(still_masked)  # Unmask all remaining on last step
            
            for i in range(num_unmask_now):
                pos, _, probs = confidences[i]
                # Sample from distribution
                sampled_token = torch.multinomial(probs, 1).item()
                x_t[0, pos] = sampled_token
                still_masked.remove(pos)
    
    # Decode result
    output_ids = x_t[0].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    
    # Verify locked spans preserved
    locked_preserved = all(
        span.lower() in output_text.lower() 
        for span in locked_spans if span.strip()
    )
    
    # Count changed tokens
    tokens_changed = sum(
        1 for pos in editable_positions 
        if output_tokens[pos] != original_tokens[pos]
    )
    
    return {
        "output": output_text,
        "locked_preserved": locked_preserved,
        "tokens_changed": tokens_changed,
        "tokens_total": len(editable_positions),
    }


@asynccontextmanager
async def lifespan(app):
    """Load model on startup."""
    # Use untrained model (trained checkpoint made it worse)
    # checkpoint = Path(__file__).parent.parent / "checkpoints" / "bert_diffusion.pt"
    # load_model(str(checkpoint) if checkpoint.exists() else None)
    load_model(None)  # Use fresh untrained timestep embedding
    yield


app = FastAPI(
    title="Constrained Diffusion LM",
    description="Text editing with constraint preservation using BertDiffusionLM",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/edit", response_model=EditResponse)
async def edit(request: EditRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = edit_text(
            text=request.text,
            locked_spans=request.locked_spans,
            temperature=request.temperature,
            num_steps=request.num_steps,
        )
        return EditResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
