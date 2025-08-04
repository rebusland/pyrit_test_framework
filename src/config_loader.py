import yaml
from dotenv import load_dotenv
from pathlib import Path
import logging
import os

# TODO get severity level from config file
LOG_LEVEL = logging.INFO

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Where to save reports, conversations and so on
OUTPUTS_DIR = "results/"

## Prepended prompts
_SYSTEM_PROMPTS_PATH = "config/system_prompts/"
EXAMPLE_SAFE_PROMPT='example_safe_prompt.txt'
CHILDREN_SAFE_PROMPT='children_safe_prompt.txt'
VERIFICATION_PROMPT='verification_prompt.txt'

load_dotenv()

### TODO MOVE TO A CLASS ###
## OPENAI CONFIGS
_AZURE_OPENAI_ENDPOINT="AZURE_OPENAI_ENDPOINT"
_AZURE_OPENAI_API_KEY="AZURE_OPENAI_API_KEY"
_AZURE_OPENAI_DEPLOYMENT="AZURE_OPENAI_DEPLOYMENT"
_AZURE_OPENAI_API_VERSION="AZURE_OPENAI_API_VERSION"

_openai_endpoint=os.getenv(_AZURE_OPENAI_ENDPOINT)
_openai_api_key=os.getenv(_AZURE_OPENAI_API_KEY)
_openai_deployment=os.getenv(_AZURE_OPENAI_DEPLOYMENT)
_openai_api_version=os.getenv(_AZURE_OPENAI_API_VERSION)

_openai_full_endpoint = f"{_openai_endpoint}/openai/deployments/{_openai_deployment}/chat/completions?api-version={_openai_api_version}"
##END OPENAI CONFIGS

### TODO MOVE to SEPARATE CLASS ###
## AZURE CONTENT FILTER API CONFIGS
_AZURE_CONTENT_FILTER_ENDPOINT="AZURE_CONTENT_FILTER_ENDPOINT"
_AZURE_CONTENT_FILTER_API_KEY="AZURE_CONTENT_FILTER_API_KEY"
_AZURE_CONTENT_FILTER_DEPLOYMENT="AZURE_CONTENT_FILTER_DEPLOYMENT"
_AZURE_CONTENT_FILTER_API_VERSION="AZURE_CONTENT_FILTER_API_VERSION"

_content_filter_endpoint=os.getenv(_AZURE_CONTENT_FILTER_ENDPOINT)
_content_filter_key=os.getenv(_AZURE_CONTENT_FILTER_API_KEY)
_content_filter_deployment=os.getenv(_AZURE_CONTENT_FILTER_DEPLOYMENT)
_content_filter_api_version=os.getenv(_AZURE_CONTENT_FILTER_API_VERSION)

_content_filter_full_endpoint = f"{_content_filter_endpoint}/openai/deployments/{_content_filter_deployment}/chat/completions?api-version={_content_filter_api_version}"
## END AZURE CONTENT FILTER API CONFIGS'''


TEST_CONFIG_PATH = "config/test_config.yaml"


def load_dotenv_with_check():
    ok_dotenv = load_dotenv()  # Load .env file into environment
    if ok_dotenv:
        logger.debug('Succesfully loaded test framework .env')
    else:
        logger.debug('Unable to load a .env file for the test framework')

def load_openai_configs():
    load_dotenv_with_check()
    ##OPENAI CONFIGS##
    _openai_endpoint=os.getenv(_AZURE_OPENAI_ENDPOINT)
    _openai_api_key=os.getenv(_AZURE_OPENAI_API_KEY)
    _openai_deployment=os.getenv(_AZURE_OPENAI_DEPLOYMENT)
    _openai_api_version=os.getenv(_AZURE_OPENAI_API_VERSION)
    _openai_full_endpoint = f"{_openai_endpoint}/openai/deployments/{_openai_deployment}/chat/completions?api-version={_openai_api_version}"

    ##AZURE CONTENT FILTER API CONFIGS##
    _content_filter_endpoint=os.getenv(_AZURE_CONTENT_FILTER_ENDPOINT)
    _content_filter_key=os.getenv(_AZURE_CONTENT_FILTER_API_KEY)
    _content_filter_deployment=os.getenv(_AZURE_CONTENT_FILTER_DEPLOYMENT)
    _content_filter_api_version=os.getenv(_AZURE_CONTENT_FILTER_API_VERSION)
    _content_filter_full_endpoint = f"{_openai_endpoint}/openai/deployments/{_openai_deployment}/chat/completions?api-version={_openai_api_version}"


def load_test_config(config_path: str = TEST_CONFIG_PATH) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_all_configs():
    load_dotenv_with_check()
    load_openai_configs()
    return load_test_config()

def load_system_prompt(prompt_name: str=EXAMPLE_SAFE_PROMPT):
    with open(_SYSTEM_PROMPTS_PATH + prompt_name) as file:
        return file.read()
