import os.path as osp
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(verbose=True)
COREF_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
