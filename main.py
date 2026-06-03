from config import SERVICES
from openai import OpenAI
from pprint import pp
import logging        
import argparse
import sys
from io import BytesIO
from PIL import Image
import base64

def _client(service: dict) -> OpenAI:
    """helper function to build the client object"""
    if not isinstance(service, dict):
        raise ValueError(f"Expected a sevice config dict, got: {type(service)}")
    return OpenAI(
        base_url=service.get("url"),
        api_key=service.get("key"),
        timeout=120.0
    )
    
def _encode_image(image_path, MAX_WIDTH=1568):
    
    img = Image.open(image_path)
    
    if img.format != "JPEG":
        img = img.convert("RGB")
    
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)))
        
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

def chat(service: dict, model: str, user_input: str, image_path: str | None = None):
    client = _client(service)
    
    content = [{"type": "text", "text": user_input}]
    
    if image_path:
        
        b64img = _encode_image(image_path)
        img_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64img}"}}
        content.append(img_content)
        
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": content
        }]
    )
    return response.choices[0].message.content

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
    parser.add_argument("--image-path", type=str, help="Include an image to the prompt")
    
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
    
    if args.prompt:
        if not args.service or not args.model:
            parser.error("--prompt requires both --service and --model")
        service_config = _validate_service(parser, args.service)
        print(chat(service_config, args.model, args.prompt, args.image_path))

if __name__ == "__main__":
    main()
