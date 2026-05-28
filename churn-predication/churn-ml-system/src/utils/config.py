import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

class ConfigLoader:
    def __init__(self, config_path: str= 'configs/config.yaml'):
        self.config_path=Path(config_path)

    def load_config(self)->dict:
        if not self.config_path.exists():
            logging.error(f'congig file is not found  at {self.config_path}')
            raise FileNotFoundError(f'config file not found at {self.config_path}')
        
        try:
            with open(self.config_path, 'r') as file:
                config=yaml.safe_load(file)
                logging.info(f'File is succesfully loaded')
                return config
        except yaml.YAMLError as e:
            logging.error(f'Error pass the yaml file')
            raise e
loader=ConfigLoader()
config=loader.load_config()
print(config)