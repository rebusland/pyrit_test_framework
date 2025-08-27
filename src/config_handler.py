import yaml
from dotenv import load_dotenv
from pathlib import Path
import os

_TEST_CONFIG = None

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

TEST_CONFIG_PATH = Path("config") / "test_config.yaml"

def load_dotenv_with_check():
    from logging_handler import logger
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

def load_test_config(config_path: Path = TEST_CONFIG_PATH) -> dict:
    with config_path.open('r') as file:
        _TEST_CONFIG = yaml.safe_load(file)
        return _TEST_CONFIG

def get_model_deployment():
    return _openai_deployment

def get_test_config(config_path: str = TEST_CONFIG_PATH) -> dict:
    return _TEST_CONFIG if _TEST_CONFIG else load_test_config(config_path=config_path)

def load_all_configs():
    load_dotenv_with_check()
    load_openai_configs()
    return load_test_config()

def load_system_prompt(prompt_name: str=get_test_config()['system_prompt']['value']):
    '''
    Remove any newline/cr characters
    '''
    sys_prompt_full_path = Path(get_test_config()['system_prompt']['dir']) / prompt_name
    with sys_prompt_full_path.open('r') as file:
        return file.read().replace('\n', '').replace('\r', '')
