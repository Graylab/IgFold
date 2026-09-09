__version__ = "1.0.1"

from .model.IgFold import IgFold
from .model.interface import IgFoldInput, IgFoldOutput
from .runner import IgFoldRunner

__all__ = ["IgFold", "IgFoldInput", "IgFoldOutput", "IgFoldRunner", "__version__"]
