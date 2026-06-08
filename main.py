import validators
from datetime import datetime
from config import SERVICES
from openai import OpenAI
from openai.types.chat import ChatCompletion 
from pprint import pp
import logging        
import argparse
import sys
from io import BytesIO
from PIL import Image
import base64
from pathlib import Path
import re
import json

Path("conversations").mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    filename='conversations/conversations.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def _client(service: dict) -> OpenAI:
    """helper function to build the client object"""
    if not isinstance(service, dict):
        raise ValueError(f"Expected a sevice config dict, got: {type(service)}")
    return OpenAI(
        base_url=service.get("url"),
        api_key=service.get("key"),
        timeout=120.0
    )

def _encode_image(image_path: str | Path, MAX_WIDTH: int = 1568):
    """Base64 encoding of attached images

    Args:
        image_path (str | Path): Path to the image to be added to the prompt message
        MAX_WIDTH (int, optional): Downsize image to keep it between the limits of the models. Defaults to 1568 to match Anthropic's recommended size. 

    Returns:
        Base64 decoded image data.
    """
    
    img = Image.open(image_path)
    
    if img.format != "JPEG":
        img = img.convert("RGB")
    
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)))
        
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def _read_if_file(value: str) -> str:
    """Read file contents if value is a valid file path, otherwise return as-is"""
    p = Path(value)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return value

def _is_image(path_obj: Path) -> bool:
    img_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return path_obj.suffix.lower() in img_extensions

def _context_window(user_input: str, image_path: str | Path | None = None, instructions: str | None = None) -> list[dict]:
    """Handles user input and shape it to OpenAI compatible message format"""
    
    messages_list = []
    
    if instructions:
        inst_content = {"role": "developer", "content": str(instructions)}
        messages_list.append(inst_content)
    
    user_content = [{"type": "text", "text": user_input}]
    
    if image_path:
        image_content = {
            "type": "image_url"
        }
        image_content["image_url"] = str(image_path) if validators.url(image_path) else f"data:image/jpeg;base64,{_encode_image(image_path)}"
        
        user_content.append(image_content)
    
    
    messages_list.append({
            "role": "user",
            "content": user_content
        })
    
    return messages_list


def list_models_from_service(service_name: str, service: dict, verbose: bool =False) -> None:
    
   client = _client(service)
   models_page = client.models.list()
   print(f"\nModels for {service_name}:")
   for model in models_page.data:
       if verbose:
           pp(model.model_dump(), indent=2)
       else:
           print(f"Model ID: {model.id}")
    

def list_models(verbose: bool =False) -> None:
    for name, config in SERVICES.items():
        if not config.get("key"):
            logging.warning(f"Skipping '{name}: missing API key")
            continue
        try:
            list_models_from_service(name, config, verbose=verbose)
        except Exception as e:
            logging.error(f"Error listing models for {name}: {e}")

def chat(service: dict, 
         model: str, 
         user_input: str, 
         image_path: str | Path | None = None, 
         instructions: str | None = None, 
         usage_data: bool = False, 
         full_response: bool = False
         ) -> (ChatCompletion | dict | str | None):
    """_summary_

    Args:
        service (dict): _description_
        model (str): _description_
        user_input (str): _description_
        image_path (str | Path | None, optional): _description_. Defaults to None.
        instructions (str | None, optional): _description_. Defaults to None.
        full_response (bool, optional): Returns the raw response from the model. Defaults to False.

    Returns:
        ChatCompletion | str | None
    """
    
    
    client = _client(service) 
    
    user_input = _read_if_file(user_input)
    instructions = _read_if_file(instructions) if instructions else None
    
    response = client.chat.completions.create(
        model=model,
        messages=_context_window(user_input, image_path, instructions) # type: ignore
    )
    
    if full_response:
        return response
    
    if usage_data:
        usage = response.usage
        return {
            "content": response.choices[0].message.content,
            "usage_data": {
                "completion_tokens": usage.completion_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "total_tokens": usage.total_tokens
            }
        }
    
    return response.choices[0].message.content

def _extract_json(response: str) -> dict | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON content found: {e}")
            return
    
    logging.warning("No json code block found")
    return

def _load_json(path):
    with open(path) as f:
        return json.load(f)

def _init_batch(service: dict, 
                     model: str, 
                     user_input: str, 
                     image_dir: str | Path, 
                     instructions: str | None,
                     conversation: str | Path | None = None) -> tuple[dict, Path]:
    
    if conversation:
        conversation_path = Path(f"conversations/{conversation}/{conversation}.json")
        if conversation_path.is_file():
            with open(conversation_path, "r") as f:
                conversation_dict = json.load(f)
            
            return conversation_dict, conversation_path
    
    # Start fresh batch
    init_batch = {
        "project_id": datetime.now().strftime('%Y%m%d%H%M%S'),
        "service_url": service.get("url"),
        "model": model,
        "source_folder": image_dir,
        "prompt": user_input,
    }
    
    if instructions:
        init_batch["instructions"] = instructions
    
    if not conversation:
        conversation = Path(f"conversations/{init_batch.get('project_id', 'batch')}")
        Path.mkdir(conversation, parents=True, exist_ok=True)
    else:
        conversation = Path(f"conversations/{conversation}")
    
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise OSError(f"The folder {image_dir} doesn't exists")
    
    if not any(img_dir.iterdir()):
        raise OSError(f"Directory {str(img_dir)} is empty.")
    
    project_filename = Path(conversation, init_batch.get('project_id', 'batch')).with_suffix(".json")
    with open(project_filename, "w", encoding="utf-8") as f:
        json.dump(init_batch, f, indent=4)
    
    init_batch["start"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return init_batch, project_filename

def _already_processed(batch_dict, conversation_id, image: Path):
    
    if batch_dict.get("results"):
        results = [result.get("image_path") for result in batch_dict.get("results")]
        if str(image) in results:
            return True
              
    return Path("conversations", conversation_id, image.stem).with_suffix(".json").exists()

def _attach_results(batch_dict):
    current_results = batch_dict.get("results", [])
    new_results = [_load_json(f) for f in Path("conversations", batch_dict["project_id"]).glob("*.json") if f.stem != batch_dict["project_id"]]

    imgs_seen = [r.get("image_path") for r in current_results]
    deduped_new = [r for r in new_results if r.get("image_path") not in imgs_seen]    

    return current_results + deduped_new

def _cleaning_folder(conversation_id):
    """_summary_

    Args:
        conversation_id (str): conversation ID or project_id
    """
    # Clean directory
    for filepath in Path("conversations", conversation_id).iterdir():
        if filepath.is_file() and filepath != Path("conversations", conversation_id, conversation_id).with_suffix(".json"):
            filepath.unlink()
    
def batch_structured(service: dict, 
                     model: str, 
                     user_input: str, 
                     image_dir: str | Path, 
                     instructions: str | None,
                     usage_data: bool = False,
                     conversation: str | Path | None = None,
                     clean_cache: bool = False
                     ) -> None:
    """Use the same prompt to iterate over multiple images in the same folder. No increment in context window"""
      
    batch_dict, project_filename = _init_batch(service, model, user_input, image_dir, instructions, conversation)
    
    # Starting the batch processing
    
    usage_data_empty = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0
    }

    usage_data_cum = batch_dict.get("usage_data", usage_data_empty)
    
    for image in Path(image_dir).iterdir():
        if not _is_image(image):
            continue
        
        if _already_processed(batch_dict, batch_dict["project_id"], image):
            logging.info(f"Image {str(image)} was processed already. Skipping...")
            continue
        
        response = chat(service, 
                        model, 
                        user_input, 
                        image_path=image, 
                        instructions=instructions, 
                        usage_data=usage_data
                        )
        
        content = response.get("content", "")
        json_response = _extract_json(content)
        if not isinstance(json_response, dict):
            preview = str(json_response)[:50]
            logging.warning(f"Response ...{preview}... is not in JSON valid format. Skipped from results")
            continue
        
        # inject image path to json_response
        json_response["image_path"] = str(image)
        
        if usage_data:
            usage_data_cum["completion_tokens"] += response["usage_data"].get("completion_tokens", 0)
            usage_data_cum["prompt_tokens"] += response["usage_data"].get("prompt_tokens", 0)
            usage_data_cum["total_tokens"] += response["usage_data"].get("total_tokens", 0)
        
        with open(Path("conversations", batch_dict["project_id"], image.stem).with_suffix(".json"), "w") as r:
            json.dump(json_response, r, indent=4, ensure_ascii=False)
            
        logging.info(f"Image {str(image)} was successfuly processed.")
            
        batch_dict["usage_data"] = usage_data_cum
        with open(project_filename, "w", encoding="utf-8") as f:
            json.dump(batch_dict, f, indent=4, ensure_ascii=False)
    
    # Attach results to main dict
    batch_dict["results"] = _attach_results(batch_dict)
    batch_dict["end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(project_filename, "w", encoding="utf-8") as f:
        json.dump(batch_dict, f, indent=4, ensure_ascii=False)
    
    if clean_cache:
        _cleaning_folder(batch_dict["project_id"]) 

def _validate_service(parser: argparse.ArgumentParser, service_name: str) -> dict:
    if service_name not in SERVICES:
        parser.error(f"Unknown service '{service_name}'. Available: {list(SERVICES)}")
    config = SERVICES[service_name]
    if not config.get("key"):
        parser.error(f"Service '{service_name}' has no API key configured")
    return config

def main():
    parser = argparse.ArgumentParser(description="Basic CLI to interact with UCSB LLM providers")
    
    imgs_args = parser.add_mutually_exclusive_group()
    list_group = parser.add_mutually_exclusive_group()
    response_group = parser.add_mutually_exclusive_group()
    
    parser.add_argument("--service", type=str, help=f"Service provider (e.g. 'grit', 'dream-lab'). " f"Available: {list(SERVICES)}")
    list_group.add_argument("--list-models", action="store_true", help="List model IDs. Scoped to --service if provided, otherwise all services.")
    list_group.add_argument("--list-models-v", action="store_true", help="Like --list-models but dumps full model metadata.")
    parser.add_argument("--model", type=str, help="Model ID to use. Requires --service.")
    parser.add_argument("--prompt", type=str, help="Prompt to send. Requires --service and --model.")
    parser.add_argument("--instructions", type=str, help="Instructions to guide the model.")
    response_group.add_argument("--usage-data", action="store_true", help="Include completion usage (completion, prompt and total tokens) with the response")
    response_group.add_argument("--raw-response", action="store_true", help="Returns the full API response")
    imgs_args.add_argument("--image-path", type=str, help="Include an image to the prompt")
    imgs_args.add_argument("--batch-structured", type=str, help="Path to a directory with images. Requires --service, --model and --prompt. Returns a JSON structured result")
    parser.add_argument("--conversation", type=str, help="ID of the conversation. For instance, 20260604145733")
    parser.add_argument("--clean-cache", action="store_true", help="remove temp results getting during batch processing")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    if args.list_models or args.list_models_v:
        verbose = args.list_models_v
        if isinstance(args.service, str):
            service_config = _validate_service(parser, args.service)
            list_models_from_service(args.service, service_config, verbose=verbose)
        else:
            list_models(verbose=verbose)
        return
    
    if args.batch_structured:
        if args.prompt:
            if not args.service or not args.model:
                parser.error("--prompt requires both --service and --model")
        service_config = _validate_service(parser, args.service)
        batch_structured(service_config, args.model, args.prompt, args.batch_structured, args.instructions,
                         args.usage_data, args.conversation, args.clean_cache)
        return
    
    if args.prompt:
        if not args.service or not args.model:
            parser.error("--prompt requires both --service and --model")
        service_config = _validate_service(parser, args.service)
        print(chat(service_config, args.model, args.prompt, args.image_path, args.instructions,
                   args.usage_data, args.raw_response))

if __name__ == "__main__":
    main()
