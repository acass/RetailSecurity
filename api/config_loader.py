import yaml
from typing import Dict, Any, List

CONFIG_PATH = "../config/config.yaml"

def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found at {CONFIG_PATH}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML configuration: {e}")

# Load configuration once and make it available for import
config = load_config()

def get_api_config() -> Dict[str, Any]:
    """Returns the API configuration."""
    return config.get('api', {})

def get_model_config() -> Dict[str, Any]:
    """Returns the model configuration."""
    return config.get('model', {})

def get_camera_config() -> List[Dict[str, Any]]:
    """Returns the camera sources configuration."""
    return config.get('cameras', [])
