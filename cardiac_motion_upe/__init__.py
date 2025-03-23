import os, sys; 

sys.path.append(PKG_DIR := os.path.dirname(os.path.realpath(__file__)))

BASE_DIR = os.path.dirname(PKG_DIR)

GWAS_RESULTS_DIR = os.getenv("GWAS_RESULTS_DIR", os.path.join(BASE_DIR, "results/gwas"))

assert os.path.exists(GWAS_RESULTS_DIR), f"{GWAS_RESULTS_DIR=} does not exist. You can set an environment variable with this name containing the correct path."

import logging
LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
logging.basicConfig(level=LOGLEVEL)

from ensemble_helpers import (
    EnsembleGWASResults    
)

from loci_helpers import (
    fetch_loci_mapping
)
