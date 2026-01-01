#!/usr/bin/env python3
"""
Gradio app with embedded React frontend.
Serves React build and provides API via Gradio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import gradio as gr
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

def load_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Load the model and initialize components."""
    global MODEL, TOKENIZER, SCHEDULE, SAMPLER, DEVICE
    
    DEVICE = get_device("auto")
    print(f"Using device: {DEVICE}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    dim = state_dict["token_embedding.weight"].shape[1]
    max_seq_len = state_dict["position_embedding.weight"].shape[0]
    num_layers = sum(1 for k in state_dict if "transformer.layers" in k and "self_attn.in_proj_weight" in k)
    num_heads = dim // 64 if dim >= 64 else dim // 32
    dim_feedforward = state_dict["transformer.layers.0.linear1.weight"].shape[0]
    
    TOKENIZER = Tokenizer("bert-base-uncased", max_length=max_seq_len)
    
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
    
    SCHEDULE = get_schedule("cosine", 100)
    SAMPLER = ImprovedConstrainedSampler(
        model=MODEL,
        schedule=SCHEDULE,
        mask_token_id=TOKENIZER.mask_token_id,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    
    print(f"Model loaded: {MODEL.get_num_params():,} parameters")
    return True

def edit_text_json(
    input_text: str,
    locked_spans: str,
    diffusion_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    edit_strength: float,
):
    """Edit text and return JSON response."""
    global MODEL, TOKENIZER, SAMPLER, DEVICE
    
    if MODEL is None:
        return json.dumps({"error": "Model not loaded"})
    
    if not input_text.strip():
        return json.dumps({"error": "Input text cannot be empty"})
    
    # Tokenize
    token_ids = TOKENIZER.encode(
        input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
    )
    x_0 = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    
    # Create lock mask
    lock_mask = torch.zeros_like(x_0, dtype=torch.bool)
    locked_texts = []
    
    if locked_spans.strip():
        for span in locked_spans.split(","):
            span = span.strip()
            if span and span.lower() in input_text.lower():
                span_lock_mask, _, found_spans = lock_substring(
                    input_text, span, TOKENIZER, seq_len=x_0.shape[1]
                )
                span_lock_tensor = span_lock_mask.unsqueeze(0).to(DEVICE)
                lock_mask = lock_mask | span_lock_tensor
                if found_spans:
                    locked_texts.append(span)
    
    # Update schedule
    global SCHEDULE
    SCHEDULE = get_schedule("cosine", diffusion_steps)
    SAMPLER.schedule = SCHEDULE
    SAMPLER.num_timesteps = diffusion_steps
    
    # Apply edit strength
    edit_mask = ~lock_mask
    if edit_strength < 1.0:
        import random
        editable_positions = edit_mask[0].nonzero(as_tuple=True)[0].tolist()
        num_to_mask = max(1, int(len(editable_positions) * edit_strength))
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
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            repetition_penalty=repetition_penalty,
        )
    
    # Decode
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
    
    for i, (orig, edit, locked) in enumerate(zip(original_tokens, edited_tokens, lock_mask_list)):
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
    
    result = {
        "generated_text": generated_text,
        "visualization": visualization,
        "metrics": {
            "constraint_fidelity": float(metrics.constraint_fidelity),
            "edit_rate": float(metrics.edit_rate),
            "num_locked_tokens": int(metrics.num_locked_tokens),
            "num_locked_preserved": int(metrics.num_locked_preserved),
        }
    }
    
    return json.dumps(result)

def create_interface():
    """Create Gradio interface with embedded React app."""
    
    # Load React build HTML
    react_build_path = Path(__file__).parent.parent / "frontend" / "dist" / "index.html"
    
    if not react_build_path.exists():
        # Show message if React app not built
        with gr.Blocks() as demo:
            gr.Markdown("""
            # React Frontend Not Built
            
            Please build the React frontend first:
            
            ```bash
            cd frontend
            npm install
            npm run build
            ```
            """)
        return demo
    
    # Serve React app via iframe from a simple HTTP server
    # We'll start a simple HTTP server on a different port for the React app
    import http.server
    import socketserver
    import threading
    import webbrowser
    
    dist_path = react_build_path.parent
    react_port = 3000
    
    # Start simple HTTP server for React app in background
    class ReactHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dist_path), **kwargs)
    
    def start_react_server():
        with socketserver.TCPServer(("", react_port), ReactHandler) as httpd:
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=start_react_server, daemon=True)
    server_thread.start()
    
    # Store the edit function globally so JavaScript can access it
    global edit_text_json_func
    edit_text_json_func = edit_text_json
    
    with gr.Blocks() as demo:
        # Hidden API components
        with gr.Row(visible=False):
            input_text_api = gr.Textbox(elem_id="api_input_text")
            locked_spans_api = gr.Textbox(elem_id="api_locked_spans")
            diffusion_steps_api = gr.Number(value=100, elem_id="api_diffusion_steps")
            temperature_api = gr.Number(value=0.8, elem_id="api_temperature")
            top_k_api = gr.Number(value=50, elem_id="api_top_k")
            top_p_api = gr.Number(value=0.9, elem_id="api_top_p")
            repetition_penalty_api = gr.Number(value=1.2, elem_id="api_repetition_penalty")
            edit_strength_api = gr.Number(value=0.4, elem_id="api_edit_strength")
            output_json_api = gr.Textbox(elem_id="api_output_json")
        
        # Connect function with button click
        api_btn = gr.Button("", visible=False, elem_id="api_trigger_btn")
        api_btn.click(
            fn=edit_text_json,
            inputs=[
                input_text_api,
                locked_spans_api,
                diffusion_steps_api,
                temperature_api,
                top_k_api,
                top_p_api,
                repetition_penalty_api,
                edit_strength_api,
            ],
            outputs=[output_json_api],
        )
        
        # Create iframe HTML
        iframe_html = f"""
        <iframe 
            src="http://localhost:{react_port}/index.html" 
            style="width: 100%; height: 100vh; border: none;"
            id="react-iframe"
        ></iframe>
        <script>
            (function() {{
                // Listen for messages from React app
                window.addEventListener('message', async function(event) {{
                    if (event.data.type === 'gradio-edit-request') {{
                        const params = event.data.params;
                        const iframe = document.getElementById('react-iframe');
                        
                        try {{
                            // Use Gradio's API directly
                            if (window.gradioApp) {{
                                const iface = window.gradioApp.getInterfaceById(0);
                                if (iface && iface.api) {{
                                    // Get the button's fn_index
                                    const btn = document.getElementById('api_trigger_btn');
                                    if (btn) {{
                                        // Set input values first
                                        const inputText = document.getElementById('api_input_text');
                                        const lockedSpans = document.getElementById('api_locked_spans');
                                        const diffusionSteps = document.getElementById('api_diffusion_steps');
                                        const temperature = document.getElementById('api_temperature');
                                        const topK = document.getElementById('api_top_k');
                                        const topP = document.getElementById('api_top_p');
                                        const repPenalty = document.getElementById('api_repetition_penalty');
                                        const editStrength = document.getElementById('api_edit_strength');
                                        
                                        if (inputText) {{
                                            inputText.value = params[0];
                                            inputText.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                        }}
                                        if (lockedSpans) {{
                                            lockedSpans.value = params[1];
                                            lockedSpans.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                        }}
                                        
                                        // Click the button to trigger
                                        btn.click();
                                        
                                        // Wait for output
                                        const outputJson = document.getElementById('api_output_json');
                                        let attempts = 0;
                                        const checkOutput = setInterval(() => {{
                                            attempts++;
                                            if (outputJson && outputJson.value) {{
                                                clearInterval(checkOutput);
                                                const result = outputJson.value;
                                                iframe.contentWindow.postMessage({{
                                                    type: 'gradio-edit-response',
                                                    data: {{ data: [result] }}
                                                }}, '*');
                                            }} else if (attempts > 300) {{
                                                clearInterval(checkOutput);
                                                iframe.contentWindow.postMessage({{
                                                    type: 'gradio-edit-response',
                                                    data: {{ error: 'Timeout waiting for response' }}
                                                }}, '*');
                                            }}
                                        }}, 100);
                                    }}
                                }}
                            }}
                        }} catch (e) {{
                            console.error('Error:', e);
                            iframe.contentWindow.postMessage({{
                                type: 'gradio-edit-response',
                                data: {{ error: e.message }}
                            }}, '*');
                        }}
                    }}
                }});
            }})();
        </script>
        """
        
        # Embed React app via iframe
        gr.HTML(value=iframe_html, elem_id="react-app")
    
    return demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    print("Loading model...")
    load_model(args.checkpoint)
    
    print("Creating Gradio interface...")
    demo = create_interface()
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )

