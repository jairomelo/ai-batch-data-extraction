import validators
import uuid
from datetime import datetime
from config import SERVICES
from openai import OpenAI
from pprint import pp
import logging        
import argparse
import sys
from io import BytesIO
from PIL import Image
import base64
from pathlib import Path

def _client(service: dict) -> OpenAI:
    """helper function to build the client object"""
    if not isinstance(service, dict):
        raise ValueError(f"Expected a sevice config dict, got: {type(service)}")
    return OpenAI(
        base_url=service.get("url"),
        api_key=service.get("key"),
        timeout=120.0
    )
    
def _conversation(prompt):    
    return {
        "id": uuid.uuid4().hex,
        "start": datetime.now(),
        "name": prompt[:100]
    }
    

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

def chat(service: dict, model: str, user_input: str, image_path: str | Path | None = None, instructions: str | None = None):
    
    client = _client(service) 
    
    user_input = _read_if_file(user_input)
    instructions = _read_if_file(instructions) if instructions else None
    
    response = client.chat.completions.create(
        model=model,
        messages=_context_window(user_input, image_path, instructions) # type: ignore
    )
    
    return response.choices[0].message.content

def batch_processing(service: dict, model: str, user_input: str, image_dir: str | Path, instructions: str | None):
    """Use the same prompt to iterate over multiple images in the same folder. No increment in context window"""
    
    conversation = _conversation(user_input)
    
    destination_directory = Path("conversations", str(conversation.id))
    
    Path.mkdir(destination_directory, parents=True, exist_ok=True)
    
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"The folder {image_dir} doesn't exists")
    
    for image in Path(image_dir).iterdir():
        response = chat(service, model, user_input, image_path=image, instructions=instructions)
        filename = Path(image.stem).with_suffix(".json")
        with open(Path(destination_directory, filename), "w") as f:
            f.write(response)
            
    return response[:500]


def _validate_service(parser: argparse.ArgumentParser, service_name: str) -> dict:
    if service_name not in SERVICES:
        parser.error(f"Unknown service '{service_name}'. Available: {list(SERVICES)}")
    config = SERVICES[service_name]
    if not config.get("key"):
        parser.error(f"Service '{service_name}' has no API key configured")
    return config

def main():
    parser = argparse.ArgumentParser(description="Basic CLI to interact with UCSB LLM providers")
    parser.add_argument("--service", type=str, help=f"Service provider (e.g. 'grit', 'dream-lab'). " f"Available: {list(SERVICES)}")
    parser.add_argument("--list-models", action="store_true", help="List model IDs. Scoped to --service if provided, otherwise all services.")
    parser.add_argument("--list-models-v", action="store_true", help="Like --list-models but dumps full model metadata.")
    parser.add_argument("--model", type=str, help="Model ID to use. Requires --service.")
    parser.add_argument("--prompt", type=str, help="Prompt to send. Requires --service and --model.")
    parser.add_argument("--instructions", type=str, help="Instructions to guide the model.")
    parser.add_argument("--image-path", type=str, help="Include an image to the prompt")
    parser.add_argument("--batch", type=str, help="Path to a directory with images. Requires --service, --model and --prompt")
    
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
    
    if args.batch:
        if args.prompt:
            if not args.service or not args.model:
                parser.error("--prompt requires both --service and --model")
        service_config = _validate_service(parser, args.service)
        batch_processing(service_config, args.model, args.prompt, args.batch, args.instructions)
    
    if args.prompt:
        if not args.service or not args.model:
            parser.error("--prompt requires both --service and --model")
        service_config = _validate_service(parser, args.service)
        print(chat(service_config, args.model, args.prompt, args.image_path, args.instructions))

if __name__ == "__main__":
    main()
