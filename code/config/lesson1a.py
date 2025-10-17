import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from llms import get_llm
from langchain_core.messages import HumanMessage

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from paths import PROMPT_CONFIG_FPATH, OUTPUTS_DIR, APP_CONFIG_FPATH
from prompt_builder import build_prompt_from_config

def invoke_llm(prompt:str,model :str='groq/compound') -> Optional[str]:
   """Calls the LLM with a prompt and returns the response.

    Args:
        prompt: The prompt to send to the LLM.
        model: The LLM model to use.
        temperature: Sampling temperature.

    Returns:
        The LLM's response content, or None if an error occurs.
    """
   try:
        llm = get_llm(model)
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
   except Exception as e:
        print(f"Error calling llm: {e}")
        return None

def run_prompt_example(
        all_prompt_config: Dict[str, Any],
        prompt_config_key: str, 
        publication_content: str, 
        model_name: str,
        app_config: Dict[str, Any]
)-> None:
    """Builds a prompt, runs it with the LLM, and saves the response.

    Args:
        all_prompts_config: Dictionary of all available prompt configurations.
        prompt_config_key: Key identifying the specific prompt config to use.
        publication_content: Content to summarize or process.
        model_name: Name of the LLM to use.
        app_config: Application-level config (e.g. reasoning strategies).
    """
    #build the prompt
    if prompt_config_key not in all_prompt_config:
        print(f"config key '{prompt_config_key}' not found in configuration.")
        return
    prompt_config = all_prompt_config[prompt_config_key]
    # Pass app_config to build_prompt_from_config
    prompt = build_prompt_from_config(prompt_config,publication_content,app_config)
    save_text_to_file(
        prompt,
        os.path.join(OUTPUTS_DIR,f"{prompt_config_key}_prompt.md"),
        header = f"Prompt generated from Config: {prompt_config_key}"

    )

    llm_response = invoke_llm(prompt,model_name)
    if llm_response:
        save_text_to_file(
            llm_response,
            os.path.join(OUTPUTS_DIR,f"{prompt_config_key}_llm_response.md"),
            header = f"LLM Response for Prompt Config: {prompt_config_key}"
        )
    else:
            print("✗ LLM response was empty or failed.")
    
def main(prompt_config_key: str) -> None:
    """Main entry point to run a modular prompt example using configuration.

    Args:
        prompt_config_key: The key of the prompt configuration to use.
    """
    try:
        print("=" * 80)
        # Load environment variables
        print("\nLoading environment variables...")
        load_env()
        print("✓ API key loaded")

        # Load the publication content
        print("Loading publication content...")
        publication_content = load_publication()
        print(f"✓ Publication loaded ({len(publication_content)} characters)")

        print("Loading application configuration...")
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm", "groq/compound")
        print(f"✓ Model set to: {model_name}")

        # Load the prompt configuration
        print(f"Loading prompt config from: {PROMPT_CONFIG_FPATH}")
        all_prompts_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        print(f"✓ Config loaded with prompt keys: {list(all_prompts_config.keys())}")

        if prompt_config_key not in all_prompts_config:
            print(f"Error: Prompt config key '{prompt_config_key}' not found.")
            return

        print(f"\nRunning prompt example: {prompt_config_key}")
        run_prompt_example(
            all_prompt_config=all_prompts_config,
            prompt_config_key=prompt_config_key,
            publication_content=publication_content,
            model_name=model_name,
            app_config=app_config,  # Pass app_config here
        )

        print(f"\n{'-'*80}")
        print("TASK COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"Error in lesson execution: {e}")
        return None


if __name__ == "__main__":

    # Define the prompt configuration key to use
    # You can change this to any key defined in your `prompt_config.yaml` file.
    prompt_cfg_key = "summarization_prompt_cfg5"

    main(prompt_config_key=prompt_cfg_key)

