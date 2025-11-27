"""
Gradio web UI for Triage Benchmark.
"""
import os
import sys
import json
from typing import Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gradio as gr

from triage_bench.benchmark_runner import run_benchmark, TRIAGE_LEVELS

# Set dummy API keys if not set (to avoid import errors)
if "KEY_OPENAI" not in os.environ:
    os.environ["KEY_OPENAI"] = "dummy-key-will-be-replaced"
if "KEY_DEEPSEEK" not in os.environ:
    os.environ["KEY_DEEPSEEK"] = "dummy-key-will-be-replaced"


def format_summary(summary: dict) -> str:
    """Format summary metrics for display."""
    lines = [
        "## 📊 Benchmark Results",
        "",
        f"**Model**: {summary['model']}",
        f"**Vignette Set**: {summary['vignette_set']}",
        f"**Runs**: {summary['runs']}",
        f"**Vignettes Evaluated**: {summary['num_vignettes']}",
        "",
        "### Overall Metrics",
        f"- **Total Predictions**: {summary['total_predictions']}",
        f"- **Overall Accuracy**: {summary['overall_accuracy']:.2%} ({summary['correct']} correct, {summary['incorrect']} incorrect)",
        f"- **Safety Rate**: {summary['safety_rate']:.2%} ({summary['safe_predictions']}/{summary['total_predictions']})",
        f"- **Over-triage Rate**: {summary['overtriage_rate']:.2%} ({summary['overtriage_errors']} errors)",
        "",
        "### Accuracy by Triage Level",
    ]
    
    for level in TRIAGE_LEVELS:
        level_data = summary['per_level_accuracy'][level]
        lines.append(
            f"- **{level.upper()}**: {level_data['accuracy']:.2%} "
            f"({level_data['correct']}/{level_data['total']})"
        )
    
    return "\n".join(lines)


def run_benchmark_with_progress(
    model: str,
    provider: str,
    custom_model: str,
    vignette_set: str,
    runs: int,
    use_full_set: bool,
    num_vignettes: int,
    openai_key: str,
    deepseek_key: str,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """
    Run benchmark with live progress updates.
    
    Returns:
        Tuple of (summary_markdown, results_json)
    """
    # Set API keys if provided
    if openai_key:
        os.environ["KEY_OPENAI"] = openai_key
    if deepseek_key:
        os.environ["KEY_DEEPSEEK"] = deepseek_key
    
    # Determine model name based on provider
    if provider == "Custom":
        model_name = custom_model
    else:
        model_name = model
    
    # Determine number of vignettes
    num_vignettes_param = None if use_full_set else num_vignettes
    
    # Progress callback
    def progress_callback(progress_dict: dict):
        completed = progress_dict["completed"]
        total = progress_dict["total"]
        run = progress_dict["run"]
        runs_total = progress_dict["runs"]
        current_case = progress_dict["current_case"]
        total_cases = progress_dict["total_cases"]
        
        progress(
            completed / total if total > 0 else 0,
            desc=f"Run {run}/{runs_total} | Case {current_case}/{total_cases} | Total: {completed}/{total}",
        )
    
    try:
        # Run benchmark
        summary, results = run_benchmark(
            model=model_name,
            vignette_set=vignette_set,
            runs=runs,
            num_vignettes=num_vignettes_param,
            progress_callback=progress_callback,
        )
        
        # Format results
        summary_markdown = format_summary(summary)
        results_json = json.dumps(results, indent=2)
        
        return summary_markdown, results_json
        
    except Exception as e:
        error_msg = f"❌ Error running benchmark: {str(e)}\n\n"
        error_msg += "**Common issues:**\n"
        error_msg += "- Check that API keys are correct (for cloud models)\n"
        error_msg += "- Ensure Ollama is running: `ollama serve` (for Ollama models)\n"
        error_msg += "- Pull the model first: `ollama pull <model-name>` (for Ollama models)\n"
        error_msg += "- Verify model name is correct\n"
        
        import traceback
        error_msg += f"\n**Error details:**\n```\n{traceback.format_exc()}\n```"
        
        return error_msg, ""


# Define model options
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4.5-preview",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
]

DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
]

OLLAMA_MODELS = [
    "llama3.1",
    "llama3",
    "llama2",
    "qwen2.5",
    "qwen2",
    "qwen2.5:72b",
    "deepseek-r1",
    "deepseek-r1-distill",
    "mistral",
    "mixtral",
    "mistral-small",
]


def create_ui():
    """Create and launch Gradio interface."""
    
    with gr.Blocks(title="Triage Benchmark", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🏥 Medical Triage Benchmark
            
            Evaluate LLM performance on medical triage classification tasks.
            Models classify clinical vignettes into Emergency (em), Non-Emergency (ne), or Self-care (sc) categories.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                # API Keys
                with gr.Accordion("🔐 API Keys (for cloud models)", open=False):
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        info="Required for OpenAI models. Leave empty if using environment variable.",
                    )
                    deepseek_key = gr.Textbox(
                        label="DeepSeek API Key",
                        type="password",
                        placeholder="...",
                        info="Required for DeepSeek models. Leave empty if using environment variable.",
                    )
                
                # Model Selection
                provider = gr.Radio(
                    choices=["OpenAI", "DeepSeek", "Ollama", "Custom"],
                    value="DeepSeek",
                    label="Model Provider",
                    info="Select the model provider",
                )
                
                model = gr.Dropdown(
                    choices=DEEPSEEK_MODELS,
                    value="deepseek-chat",
                    label="Model",
                    visible=True,
                    info="Select the model to evaluate",
                )
                
                custom_model = gr.Textbox(
                    label="Custom Model Name",
                    placeholder="e.g., ollama+llama3.1, http://localhost:5013",
                    visible=False,
                    info="Enter any model name or URL",
                )
                
                # Update model dropdown based on provider
                def update_model_visibility(provider_choice):
                    if provider_choice == "OpenAI":
                        return gr.Dropdown(choices=OPENAI_MODELS, value="gpt-4o", visible=True), gr.Textbox(visible=False)
                    elif provider_choice == "DeepSeek":
                        return gr.Dropdown(choices=DEEPSEEK_MODELS, value="deepseek-chat", visible=True), gr.Textbox(visible=False)
                    elif provider_choice == "Ollama":
                        return gr.Dropdown(choices=OLLAMA_MODELS, value="llama3.1", visible=True), gr.Textbox(visible=False)
                    else:  # Custom
                        return gr.Dropdown(visible=False), gr.Textbox(visible=True)
                
                provider.change(
                    update_model_visibility,
                    inputs=provider,
                    outputs=[model, custom_model],
                )
                
                # Benchmark Settings
                vignette_set = gr.Radio(
                    choices=["semigran", "kopka"],
                    value="semigran",
                    label="Vignette Set",
                    info="Select the vignette dataset",
                )
                
                runs = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Runs",
                    info="Number of stochastic passes per vignette",
                )
                
                use_full_set = gr.Checkbox(
                    value=True,
                    label="Use Full Vignette Set",
                    info="Uncheck to limit number of vignettes",
                )
                
                num_vignettes = gr.Number(
                    minimum=1,
                    maximum=100,
                    value=10,
                    label="Number of Vignettes",
                    info="Only used if 'Use Full Vignette Set' is unchecked",
                    visible=False,
                )
                
                def update_vignette_visibility(use_full):
                    return gr.Number(visible=not use_full)
                
                use_full_set.change(
                    update_vignette_visibility,
                    inputs=use_full_set,
                    outputs=num_vignettes,
                )
                
                run_button = gr.Button("🚀 Run Benchmark", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                results_display = gr.Markdown(
                    value="Configure settings and click 'Run Benchmark' to start evaluation.",
                )
                
                results_json = gr.Code(
                    label="Results JSON (for download)",
                    language="json",
                    visible=False,
                )
        
        # Instructions
        with gr.Accordion("ℹ️ Instructions", open=False):
            gr.Markdown(
                """
                ### How to Use
                
                1. **Configure API Keys**: If using cloud models (OpenAI, DeepSeek), enter your API keys in the API Keys section.
                   - You can also set them as environment variables: `export KEY_OPENAI="sk-..."` and `export KEY_DEEPSEEK="..."`
                
                2. **Select Model**: Choose your model provider and specific model from the dropdown.
                   - **OpenAI**: Requires OpenAI API key
                   - **DeepSeek**: Requires DeepSeek API key
                   - **Ollama**: Requires Ollama installed and running (`ollama serve`)
                   - **Custom**: Enter any model name or URL (e.g., `ollama+llama3.1`, `http://localhost:5013`)
                
                3. **Configure Benchmark**:
                   - Select vignette set (semigran or kopka)
                   - Choose number of runs (1-5)
                   - Select number of vignettes or use full set
                
                4. **Run**: Click "Run Benchmark" to start evaluation
                
                5. **View Results**: See live progress and final summary metrics
                
                ### Ollama Setup (for open-source models)
                
                If using Ollama models:
                1. Install Ollama using instructions in README.md file
                2. Pull the model: `ollama pull llama3.1` (or your chosen model)
                3. Ensure Ollama is running: `ollama serve` (usually runs automatically)
                
                ### Supported Models
                
                - **OpenAI**: GPT-4o, GPT-4.5, O1, O3 series
                - **DeepSeek**: deepseek-chat, deepseek-reasoner
                - **Ollama**: Llama, Qwen, DeepSeek R1, Mistral, and any other Ollama-compatible models
                """
            )
        
        # Run button handler
        run_button.click(
            fn=run_benchmark_with_progress,
            inputs=[
                model,
                provider,
                custom_model,
                vignette_set,
                runs,
                use_full_set,
                num_vignettes,
                openai_key,
                deepseek_key,
            ],
            outputs=[results_display, results_json],
        ).then(
            lambda: gr.Code(visible=True),
            outputs=results_json,
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)

