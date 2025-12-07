# Save this as "download_test_data.py" (overwrite the old one)
from datasets import load_dataset
from src.utils.logger import logger

# --- List of all language pairs your project needs THAT ARE AVAILABLE ---
# NOTE: The CoVoST 2 dataset does not include 'en_es' or 'en_fr'.
# We will download the 4 available pairs from your spec.
LANG_PAIRS = [
    "es_en",  # Spanish to English
    "fr_en",  # French to English
    "de_en",  # German to English
    "en_de"   # English to German
]

logger.info("Starting download of all required CoVoST 2 test sets...")
logger.info("This will happen one by one. This may take some time.")
logger.info("NOTE: 'en_es' and 'en_fr' are not part of the CoVoST 2 dataset. Skipping them.")


for pair in LANG_PAIRS:
    try:
        logger.info(f"--- Downloading test set for: {pair} ---")
        
        # --- THIS BLOCK IS NOW FIXED ---
        # 1. Using the PARQUET MIRROR "fixie-ai/covost2"
        # 2. This dataset does not require custom scripts
        dataset = load_dataset(
            "fixie-ai/covost2", # This is the community-fixed Parquet version
            pair,
            split="test"
        )
        # --- END OF FIX ---
        
        logger.info(f"✓ Successfully downloaded and cached '{pair}'.")
        
        example = dataset[0]
        # The column names are 'file', 'sentence', and 'translation'
        logger.info(f"Example audio file path: {example['file']}")
        logger.info(f"Example source text: {example['sentence']}")
        logger.info(f"Example translation: {example['translation']}\n")

    except Exception as e:
        logger.error(f"!!! FAILED to download '{pair}'. Error: {e}\n")

logger.info("--- All dataset downloads complete. ---")