# Save this as: test_mt.py
from src.mt.helsinki_mt import HelsinkiMT
from src.utils.logger import logger

def main():
    # Test 1: English to Spanish
    logger.info("--- Testing English -> Spanish ---")
    mt_en_es = HelsinkiMT(source_lang="en", target_lang="es")
    
    text_en = "Hello, I am testing the translation system."
    result = mt_en_es.translate_text(text_en, is_final=True)
    
    print(f"Original:   {text_en}")
    print(f"Translated: {result.text}")
    print("-" * 30)

    # Test 2: Spanish to English (Vice Versa Check!)
    logger.info("--- Testing Spanish -> English ---")
    # Note: We simply swap the languages here
    mt_es_en = HelsinkiMT(source_lang="es", target_lang="en")
    
    text_es = "Hola, esto es una prueba de latencia."
    result = mt_es_en.translate_text(text_es, is_final=True)
    
    print(f"Original:   {text_es}")
    print(f"Translated: {result.text}")

if __name__ == "__main__":
    main()