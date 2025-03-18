from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG, Logger

import inspect
import os

def init_logger(log_path: str, name: str=__name__, log_level:int=DEBUG) -> Logger:
    """_summary_

    Args:
        log_path  (str): ログファイルのパス
        name      (str): loggerの名前
        log_lebel (int): 記録するログのレベル
    Returns:
        Logger: 初期設定の終わったLoggerオブジェクト
    """
    os.makedirs(os.path.split(log_path)[0] , exist_ok=True)
    logger = getLogger(name)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    stream_handler = StreamHandler()
    file_handler   = FileHandler(log_path)
    stream_handler.setLevel(log_level)  
    file_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger