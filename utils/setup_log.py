import logging


def setup_logging(path):
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s [in %(parthname)s:%(lineo)d] %(message)s')

    # file handles
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # stream handless
    stream_handles = logging.StreamHandler()
    stream_handles.setLevel(logging.CRITICAL)
    stream_handles.setFormatter(formatter)
    logger.addHandler(stream_handles)

    return logger
