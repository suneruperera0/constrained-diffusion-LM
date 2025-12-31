#!/usr/bin/env python3
"""
Gradio Interface for ConstrainedDiffusionLM.

Demonstrates constraint-preserving diffusion generation in an interactive web UI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import gradio as gr

from constrained_diffusion_lm.data import Tokenizer
from constrained_diffusion_lm.data.constraints import lock_substring
from constrained_diffusion_lm.diffusion import get_schedule
from constrained_diffusion_lm.diffusion.sampler import ImprovedConstrainedSampler
from constrained_diffusion_lm.models import TransformerDenoiser
from constrained_diffusion_lm.eval.edit_metrics import compute_edit_metrics
from constrained_diffusion_lm.utils.seed import get_device


# Global model components (loaded once)
MODEL = None
TOKENIZER = None
SCHEDULE = None
SAMPLER = None
DEVICE = None


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


def edit_text(
    input_text: str,
    locked_spans: str,
    diffusion_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    edit_strength: float = 1.0,
):
    """
    Edit text while preserving locked spans.
    
    Args:
        input_text: The text to edit
        locked_spans: Comma-separated substrings to lock
        diffusion_steps: Number of denoising steps
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        Tuple of (generated_text, constraint_report, visualization)
    """
    global MODEL, TOKENIZER, SAMPLER, DEVICE
    
    if MODEL is None:
        return "❌ Model not loaded. Please wait...", "", ""
    
    if not input_text.strip():
        return "❌ Please enter some text.", "", ""
    
    # Tokenize input
    token_ids = TOKENIZER.encode(
        input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
    )
    x_0 = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    
    # Create lock mask from locked spans
    lock_mask = torch.zeros_like(x_0, dtype=torch.bool)
    
    locked_texts = []
    if locked_spans.strip():
        for span in locked_spans.split(","):
            span = span.strip()
            if span and span.lower() in input_text.lower():
                # lock_substring returns (lock_mask, edit_mask, spans)
                span_lock_mask, _, found_spans = lock_substring(input_text, span, TOKENIZER, seq_len=x_0.shape[1])
                # Convert to tensor and move to device
                span_lock_tensor = span_lock_mask.unsqueeze(0).to(DEVICE)
                lock_mask = lock_mask | span_lock_tensor
                if found_spans:
                    locked_texts.append(span)
    
    # Update schedule with requested steps
    global SCHEDULE
    SCHEDULE = get_schedule("cosine", diffusion_steps)
    SAMPLER.schedule = SCHEDULE
    SAMPLER.num_timesteps = diffusion_steps
    
    # Apply partial masking based on edit_strength
    # edit_strength=1.0 means mask all editable tokens (full regeneration)
    # edit_strength=0.3 means only mask 30% of editable tokens (preserve more context)
    edit_mask = ~lock_mask  # Positions that CAN be edited
    
    if edit_strength < 1.0:
        # Randomly select which editable positions to actually mask
        import random
        editable_positions = edit_mask[0].nonzero(as_tuple=True)[0].tolist()
        num_to_mask = max(1, int(len(editable_positions) * edit_strength))
        positions_to_mask = random.sample(editable_positions, num_to_mask)
        
        # Create a new edit mask with only selected positions
        partial_edit_mask = torch.zeros_like(edit_mask)
        for pos in positions_to_mask:
            partial_edit_mask[0, pos] = True
        
        # Update lock_mask to lock the positions we're NOT editing
        lock_mask = ~partial_edit_mask
    
    # Run editing
    with torch.no_grad():
        edited_ids = SAMPLER.edit(
            x_0=x_0,
            lock_mask=lock_mask,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            repetition_penalty=repetition_penalty,
        )
    
    # Decode output
    generated_text = TOKENIZER.decode(edited_ids[0].tolist())
    
    # Compute metrics
    metrics = compute_edit_metrics(
        original_ids=x_0,
        edited_ids=edited_ids,
        lock_mask=lock_mask,
    )
    
    # Create constraint report
    constraint_status = "✅ PASSED" if metrics.constraint_fidelity == 1.0 else "❌ FAILED"
    
    report = f"""
## Constraint Validation: {constraint_status}

| Metric | Value |
|--------|-------|
| **Constraint Fidelity** | {metrics.constraint_fidelity*100:.1f}% |
| **Edit Rate** | {metrics.edit_rate*100:.1f}% |
| **Locked Tokens** | {metrics.num_locked_tokens} |
| **Locked Preserved** | {metrics.num_locked_preserved}/{metrics.num_locked_tokens} |
"""
    
    if locked_texts:
        report += f"\n**Locked Spans:** {', '.join(f'`{t}`' for t in locked_texts)}"
    
    # Create visualization
    # Access the underlying HuggingFace tokenizer for token conversion
    hf_tokenizer = TOKENIZER.tokenizer
    original_tokens = hf_tokenizer.convert_ids_to_tokens(x_0[0].tolist())
    edited_tokens = hf_tokenizer.convert_ids_to_tokens(edited_ids[0].tolist())
    
    # Build colored visualization
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
            # Green for locked (preserved)
            viz_parts.append(f'<span style="background-color: #90EE90; padding: 2px 4px; margin: 1px; border-radius: 3px;">{edit}</span>')
        elif orig != edit:
            # Red for changed
            viz_parts.append(f'<span style="background-color: #FFB6C1; padding: 2px 4px; margin: 1px; border-radius: 3px;">{edit}</span>')
        else:
            # Yellow for unchanged editable
            viz_parts.append(f'<span style="background-color: #FFFACD; padding: 2px 4px; margin: 1px; border-radius: 3px;">{edit}</span>')
    
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
    
    return generated_text, report, visualization


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="ConstrainedDiffusionLM",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        css="""
        .gradio-container { max-width: 900px !important; }
        .output-text { font-size: 18px !important; }
        """
    ) as demo:
        gr.Markdown("""
        # 🔒 Constrained Diffusion Language Model
        
        **Edit text while preserving locked spans exactly.**
        
        This demo shows constraint-preserving diffusion: locked tokens remain invariant 
        across all denoising steps, while editable regions are regenerated.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 Input Text",
                    placeholder="Enter the text you want to edit...",
                    lines=3,
                    value="The contract is governed by Ontario law and must be signed by both parties."
                )
                
                locked_spans = gr.Textbox(
                    label="🔒 Locked Spans (comma-separated)",
                    placeholder="Ontario law, both parties",
                    value="Ontario law",
                    info="Text segments that must remain unchanged"
                )
                
                with gr.Row():
                    diffusion_steps = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=100,
                        step=10,
                        label="⏱️ Diffusion Steps",
                        info="More steps = better quality, slower"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Higher = more random"
                    )
                
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Top-K",
                        info="0 = disabled"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-P (Nucleus)",
                        info="1.0 = disabled"
                    )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="🔄 Repetition Penalty",
                    info="Higher = less repetition"
                )
                
                edit_strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="✏️ Edit Strength",
                    info="Lower = preserve more original text (recommended: 0.3-0.5)"
                )
                
                generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                output_text = gr.Textbox(
                    label="📤 Generated Text",
                    lines=3,
                    elem_classes=["output-text"]
                )
                
                visualization = gr.HTML(
                    label="🎨 Token Visualization"
                )
                
                constraint_report = gr.Markdown(
                    label="📊 Constraint Report"
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["The contract is governed by Ontario law and must be signed by both parties.", "Ontario law", 100, 0.8],
                ["Please review the document at your earliest convenience.", "review the document", 100, 0.7],
                ["All disputes shall be resolved through arbitration.", "arbitration", 100, 0.9],
                ["The deadline is March 15th for all submissions.", "March 15th", 100, 0.8],
                ["Payment is due within thirty days of invoice date.", "thirty days", 100, 0.8],
            ],
            inputs=[input_text, locked_spans, diffusion_steps, temperature],
            label="📚 Example Inputs"
        )
        
        # Connect button
        generate_btn.click(
            fn=edit_text,
            inputs=[
                input_text,
                locked_spans,
                diffusion_steps,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                edit_strength,
            ],
            outputs=[output_text, constraint_report, visualization],
        )
        
        gr.Markdown("""
        ---
        ### How It Works
        
        1. **Input** → Text is tokenized into subword tokens
        2. **Lock** → Specified spans are marked as immutable  
        3. **Mask** → Editable tokens are replaced with `[MASK]`
        4. **Denoise** → Model iteratively predicts clean tokens
        5. **Clamp** → Locked tokens are enforced at every step
        6. **Output** → Final text with constraints preserved exactly
        
        *Built with 🧠 Transformer + 🎲 Diffusion + 🔒 Hard Constraints*
        """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    print("Loading model...")
    load_model(args.checkpoint)
    
    print("Starting Gradio interface...")
    demo = create_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )

