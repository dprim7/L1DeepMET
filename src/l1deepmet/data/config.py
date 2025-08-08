from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path
import yaml # type: ignore

@dataclass
class SampleConfig:
    name: str
    path: Path
    n_events: Optional[int] = None
    weight: float = 1.0

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sample path does not exist: {self.path}")

@dataclass
class DatasetConfig:
    data_dirs: List[Union[str, Path]]
    sample_names: List[str]

    feature_columns: Optional[List[str]] = None
    label_column: str = "label"
    weight_column: Optional[str] = None

    train_fraction: float = 0.8
    validation_fraction: float = 0.10
    test_fraction: float = 0.10  

    normalize: bool = True
    
    def __post_init__(self):
        '''validating config after creation'''
        self.data_dirs = [Path(d) for d in self.data_dirs]

        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test fractions must sum to 1.0, got {total}")
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        

        @classmethod
        def from_yaml(cls, yaml_path: Union[str, Path]) -> 'DatasetConfig':
            pass #TODO: finish this method