# Save this file as: src/utils/logger.py
import os
from loguru import logger

# Remove default handler
logger.remove()

# Create logs directory if not exists
os.makedirs('logs', exist_ok=True)

# Add file handler
logger.add(
    'logs/s2st_{time:YYYY-MM-DD}.log',
    format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}',
    level='INFO'
)

# Add console handler
logger.add(
    lambda msg: print(msg, end=''),
    format='{time:HH:mm:ss} | {level: <8} | {message}',
    level='INFO'
)