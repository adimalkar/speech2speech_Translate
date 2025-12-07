# Save this file as: main.py

import time
import sys
import argparse
from src.pipeline.orchestrator import LiveTranslator
from src.utils.logger import logger

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Real-Time Speech-to-Speech Translator")
    parser.add_argument("--source", type=str, default="en", help="Source language code (e.g., en, es)")
    parser.add_argument("--target", type=str, default="es", help="Target language code (e.g., es, en)")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("   REAL-TIME SPEECH-TO-SPEECH TRANSLATOR")
    logger.info("==================================================")
    logger.info(f"Direction: {args.source.upper()} (You) -> {args.target.upper()} (AI)")
    
    # 2. Initialize Pipeline with Arguments
    # We need to modify the Orchestrator to accept these!
    # (See Step 2 below)
    pipeline = LiveTranslator(source_lang=args.source, target_lang=args.target)
    
    try:
        pipeline.start()
        
        # Keep main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down system...")
        pipeline.stop()
        logger.info("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()