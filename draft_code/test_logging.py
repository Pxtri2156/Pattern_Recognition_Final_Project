import logging
logging.basicConfig(filename='./log_filename.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.debug('This is a debug log message.')
logging.info('This is a info log message.')
logging.warning('This is a warning log message.')
logging.error('This is a error log message.')
logging.critical('This is a critical log message.')