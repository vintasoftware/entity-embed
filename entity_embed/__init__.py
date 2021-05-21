"""Top-level package for entity-embed."""
import logging

# libgomp issue, must import n2 before torch. See: https://github.com/kakao/n2/issues/42
import n2  # noqa: F401

from .data_modules import *  # noqa: F401, F403
from .data_utils.field_config_parser import FieldConfigDictParser  # noqa: F401
from .data_utils.numericalizer import default_tokenizer  # noqa: F401
from .entity_embed import *  # noqa: F401, F403
from .indexes import *  # noqa: F401, F403

__author__ = "Flávio Juvenal (Vinta Software)"
__email__ = "flavio@vinta.com.br"
__version__ = "0.0.5"

# Good practice: https://docs.python-guide.org/writing/logging/#logging-in-a-library
logging.getLogger(__name__).addHandler(logging.NullHandler())
