import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class NewsLogger:
    """统一的日志配置类"""
    
    _instance: Optional['NewsLogger'] = None
    _logger: Optional[logging.Logger] = None
    _debug_mode: bool = False
    
    def __new__(cls, debug: bool = False) -> 'NewsLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, debug: bool = False):
        if self._logger is None:
            self._debug_mode = debug
            self._setup_logger(debug)
        elif debug != self._debug_mode:
            # 如果调试模式发生变化，更新设置
            self._debug_mode = debug
            self.set_debug_mode(debug)
    
    def _setup_logger(self, debug: bool = False) -> None:
        """设置日志配置"""
        self._logger = logging.getLogger('news_podcast')
        
        # 避免重复添加处理器
        if self._logger.handlers:
            return
        
        # 设置日志级别
        level = logging.DEBUG if debug else logging.INFO
        self._logger.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # 文件处理器
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'news_podcast_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件中记录所有级别的日志
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
    
    def set_debug_mode(self, debug: bool = True) -> None:
        """设置调试模式"""
        if self._logger:
            level = logging.DEBUG if debug else logging.INFO
            self._logger.setLevel(level)
            for handler in self._logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    handler.setLevel(level)
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        if self._logger is None:
            self._setup_logger(self._debug_mode)
        return self._logger
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self.get_logger()


# 全局日志实例
_news_logger = NewsLogger()


def get_logger() -> logging.Logger:
    """获取日志记录器实例"""
    return _news_logger.logger


def set_debug_mode(debug: bool = True) -> None:
    """设置调试模式"""
    _news_logger.set_debug_mode(debug)


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """记录异常信息"""
    logger.error(f"{message}: {type(exc).__name__}: {str(exc)}", exc_info=True)


def log_network_request(logger: logging.Logger, method: str, url: str, status_code: Optional[int] = None) -> None:
    """记录网络请求信息"""
    if status_code:
        logger.info(f"网络请求 {method} {url} - 状态码: {status_code}")
    else:
        logger.info(f"发起网络请求 {method} {url}")


def log_file_operation(logger: logging.Logger, operation: str, file_path: str, success: bool = True) -> None:
    """记录文件操作信息"""
    status = "成功" if success else "失败"
    logger.info(f"文件操作 {operation} {file_path} - {status}")