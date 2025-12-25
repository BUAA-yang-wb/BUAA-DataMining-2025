"""
训练日志记录器
提供详细的训练过程记录功能
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称，如果为None则使用时间戳
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志文件名
        self.log_file = os.path.join(log_dir, f"{self.experiment_name}.log")

        # 配置日志
        self._setup_logger()

        # 记录实验开始
        self.logger.info("=" * 60)
        self.logger.info("多模态异常检测训练日志")
        self.logger.info(f"实验名称: {self.experiment_name}")
        self.logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

    def _setup_logger(self):
        """设置日志配置"""
        # 创建logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if self.logger.handlers:
            return

        # 创建文件handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_config(self, config_dict: Dict[str, Any]):
        """记录配置信息"""
        self.logger.info("📋 实验配置:")
        for key, value in config_dict.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.logger.info(f"Epoch {epoch + 1}/{total_epochs} 开始")

    def log_batch_result(self, epoch: int, batch_idx: int, total_batches: int,
                        loss: float, task_type: str = "unknown", **kwargs):
        """记录批次结果"""
        progress = f"[{batch_idx + 1}/{total_batches}]"
        loss_str = ".4f"

        extra_info = ""
        if kwargs:
            extra_info = " | " + " | ".join([f"{k}: {v:.4f}" for k, v in kwargs.items()])

        self.logger.info(f"  {task_type}批次 {progress} - 损失: {loss_str}{extra_info}")

    def log_epoch_summary(self, epoch: int, avg_loss: float, lr: float = None, **metrics):
        """记录epoch总结"""
        summary = ".4f"
        if lr is not None:
            summary += ".6f"
        if metrics:
            summary += " | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        self.logger.info(f"Epoch {epoch + 1} 完成 - {summary}")
        self.logger.info("")

    def log_task_start(self, task_name: str, description: str = ""):
        """记录任务开始"""
        desc = f" - {description}" if description else ""
        self.logger.info(f"开始{task_name}{desc}")

    def log_task_end(self, task_name: str, duration: float = None, **results):
        """记录任务结束"""
        duration_str = ".2f" if duration else ""
        results_str = ""
        if results:
            formatted_results = []
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    formatted_results.append(f"{k}: {v:.4f}")
                else:
                    formatted_results.append(f"{k}: {v}")
            results_str = " | " + " | ".join(formatted_results)

        self.logger.info(f"{task_name}完成{duration_str}{results_str}")

    def log_evaluation_results(self, task_name: str, metrics: Dict[str, float]):
        """记录评估结果"""
        self.logger.info(f"{task_name} 评估结果:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")
        self.logger.info("")

    def log_error(self, error_msg: str, exc_info: bool = True):
        """记录错误信息"""
        self.logger.error(f"错误: {error_msg}", exc_info=exc_info)

    def log_warning(self, warning_msg: str):
        """记录警告信息"""
        self.logger.warning(f"警告: {warning_msg}")

    def log_info(self, info_msg: str):
        """记录一般信息"""
        self.logger.info(f"信息: {info_msg}")

    def close(self):
        """关闭日志记录器"""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.info("日志记录结束")


class BatchTimer:
    """批次计时器"""

    def __init__(self):
        self.start_time = None
        self.batch_times = []

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def lap(self):
        """记录一个批次的耗时"""
        if self.start_time is None:
            return 0.0

        current_time = time.time()
        batch_time = current_time - self.start_time
        self.batch_times.append(batch_time)
        self.start_time = current_time

        return batch_time

    def get_average_time(self):
        """获取平均批次耗时"""
        return sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0.0

    def reset(self):
        """重置计时器"""
        self.start_time = None
        self.batch_times = []


# 全局日志记录器实例
_global_logger = None


def get_logger(log_dir: str = "logs", experiment_name: Optional[str] = None) -> TrainingLogger:
    """获取全局日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger(log_dir, experiment_name)
    return _global_logger


def setup_experiment_logging(log_dir: str = "logs", experiment_name: Optional[str] = None) -> TrainingLogger:
    """设置实验日志记录"""
    return get_logger(log_dir, experiment_name)


if __name__ == "__main__":
    # 测试日志记录器
    logger = TrainingLogger("test_logs", "test_experiment")

    logger.log_config({"batch_size": 8, "epochs": 20, "lr": 1e-4})

    for epoch in range(3):
        logger.log_epoch_start(epoch, 3)

        for batch in range(5):
            loss = 0.5 - batch * 0.1 + epoch * 0.05
            logger.log_batch_result(epoch, batch, 5, loss, "训练")

        logger.log_epoch_summary(epoch, 0.3 - epoch * 0.1, lr=1e-4)

    logger.log_evaluation_results("Task2", {"auc": 0.85, "f1": 0.82})
    logger.close()

    print("日志测试完成！")
