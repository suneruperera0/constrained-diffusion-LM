"""
FastAPI backend for Constrained Diffusion LM.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.data import Tokenizer
from constrained_diffusion_lm.diffusion import get_schedule


# Global model state
model = None
tokenizer = None
device = None


class EditRequest(BaseModel):
    text: str
    locked_spans: List[str] = []
    temperature: float = 0.7
    num_steps: int = 50


class EditResponse(BaseModel):
    output: str
    locked_preserved: bool
    tokens_changed: int
    tokens_total: int


def load_model(checkpoint_path: Optional[str] = None):
    """Load model from checkpoint or pretrained BERT."""
    global model, tokenizer, device
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = Tokenizer("bert-base-uncased")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get model config
        model_config = checkpoint.get("model_config", {})
        model = TransformerDenoiser(
            vocab_size=tokenizer.vocab_size,
            dim=model_config.get("dim", 768),
            num_heads=model_config.get("num_heads", 12),
            num_layers=model_config.get("num_layers", 12),
            dim_feedforward=model_config.get("dim_feedforward", 3072),
            dropout=model_config.get("dropout", 0.1),
            max_seq_len=model_config.get("max_seq_len", 256),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Loading pretrained BERT...")
        model = TransformerDenoiser.from_pretrained_bert("bert-base-uncased")
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")


def edit_text(text: str, locked_spans: List[str], temperature: float = 0.7) -> dict:
    """Edit text while preserving locked spans."""
    global model, tokenizer, device
    
    # Tokenize
    encoded = tokenizer.encode(text, add_special_tokens=True)
    x = torch.tensor([encoded], device=device)
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(encoded)
    original_tokens = tokens.copy()
    
    # Create lock mask
    lock = [False] * len(tokens)
    lock[0] = lock[-1] = True  # [CLS] and [SEP]
    
    for span in locked_spans:
        span_toks = tokenizer.tokenizer.convert_ids_to_tokens(
            tokenizer.encode(span, add_special_tokens=False)
        )
        for i in range(len(tokens) - len(span_toks) + 1):
            if tokens[i:i+len(span_toks)] == span_toks:
                for j in range(len(span_toks)):
                    lock[i+j] = True
    
    # Mask non-locked tokens
    x_masked = x.clone()
    editable_positions = []
    for i, locked in enumerate(lock):
        if not locked:
            x_masked[0, i] = tokenizer.mask_token_id
            editable_positions.append(i)
    
    # Predict
    with torch.no_grad():
        t = torch.tensor([0], device=device)
        logits = model(x_masked, t, torch.ones_like(x_masked))
        
        result = x_masked.clone()
        for i in editable_positions:
            probs = torch.softmax(logits[0, i] / temperature, dim=-1)
            result[0, i] = torch.multinomial(probs, 1)
    
    output_text = tokenizer.decode(result[0].tolist())
    output_tokens = tokenizer.tokenizer.convert_ids_to_tokens(result[0].tolist())
    
    # Check locked spans preserved
    locked_preserved = all(span.lower() in output_text.lower() for span in locked_spans)
    
    # Count changed tokens
    tokens_changed = sum(1 for i in editable_positions if output_tokens[i] != original_tokens[i])
    
    return {
        "output": output_text,
        "locked_preserved": locked_preserved,
        "tokens_changed": tokens_changed,
        "tokens_total": len(editable_positions),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    checkpoint = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
    load_model(str(checkpoint) if checkpoint.exists() else None)
    yield


app = FastAPI(
    title="Constrained Diffusion LM",
    description="Text editing with constraint preservation",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
        )
        return EditResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
