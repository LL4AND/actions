import threading
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

from ..common.logging import logger
from .backup_service import BackupService
from ..configs.config import Config

class AutoBackupManager:
    """Manages automatic backups during training process"""
    
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config=None):
        with self._lock:
            if not self._initialized:
                self.config = config or Config.from_env()
                self.backup_service = BackupService(self.config)
                self.backup_thread = None
                self.stop_flag = threading.Event()
                self.thread_lock = threading.Lock()
                self.backup_interval = int(self.config.get("AUTO_BACKUP_INTERVAL_MINUTES", "30"))
                self.max_auto_backups = int(self.config.get("MAX_AUTO_BACKUPS", "5"))
                self.auto_backup_enabled = self.config.get("AUTO_BACKUP_ENABLED", "true").lower() == "true"
                self.max_retries = int(self.config.get("BACKUP_MAX_RETRIES", "3"))
                self.retry_delay = int(self.config.get("BACKUP_RETRY_DELAY_SECONDS", "60"))
                self._initialized = True
    
    def create_pre_training_backup(self, model_name=None):
        """Creates a backup before training starts"""
        if not self.auto_backup_enabled:
            logger.info("Auto backup is disabled, skipping pre-training backup")
            return None
            
        description = f"Pre-training automatic backup{' for ' + model_name if model_name else ''}"
        logger.info(f"Creating pre-training backup: {description}")
        return self.backup_service.create_backup(description=description)
    
    def start_periodic_backup(self, model_name=None):
        """Starts a thread that creates periodic backups during training"""
        if not self.auto_backup_enabled:
            logger.info("Auto backup is disabled, not starting periodic backups")
            return
            
        # Stop any existing backup thread
        self.stop_periodic_backup()
        
        # Reset stop flag
        self.stop_flag.clear()
        
        # Start new backup thread
        self.backup_thread = threading.Thread(
            target=self._periodic_backup_worker,
            args=(model_name,),
            daemon=True
        )
        self.backup_thread.start()
        logger.info(f"Started periodic backup thread with interval {self.backup_interval} minutes")
    
    def stop_periodic_backup(self):
        """Stops the periodic backup thread if it's running"""
        if self.backup_thread and self.backup_thread.is_alive():
            self.stop_flag.set()
            self.backup_thread.join(timeout=5)
            logger.info("Stopped periodic backup thread")
    
    def _periodic_backup_worker(self, model_name=None):
        """Worker function that creates periodic backups"""
        last_backup_time = None
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while not self.stop_flag.is_set():
            try:
                current_time = datetime.now()
                
                # Check if enough time has passed since last backup
                if last_backup_time and (current_time - last_backup_time).total_seconds() < self.backup_interval * 60:
                    time.sleep(60)  # Check every minute
                    continue
                
                # Create backup with retries
                description = f"Training in-progress automatic backup{' for ' + model_name if model_name else ''}"
                logger.info(f"Creating periodic backup: {description}")
                
                retry_count = 0
                backup_success = False
                base_delay = self.retry_delay
                
                while retry_count < self.max_retries:
                    try:
                        backup_result = self.backup_service.create_backup(
                            description=description,
                            tags=["auto", "training", model_name] if model_name else ["auto", "training"]
                        )
                        if backup_result:
                            logger.info("Backup created successfully")
                            last_backup_time = current_time
                            consecutive_failures = 0
                            backup_success = True
                            break
                        raise Exception("Backup creation failed")
                    except Exception as backup_error:
                        retry_count += 1
                        error_msg = str(backup_error)
                        if retry_count < self.max_retries:
                            # 使用指数退避策略计算下一次重试延迟
                            current_delay = min(base_delay * (2 ** (retry_count - 1)), 300)  # 最大延迟5分钟
                            jitter = random.uniform(0, min(current_delay * 0.1, 30))  # 添加随机抖动
                            retry_delay = current_delay + jitter
                            logger.warning(f"Backup attempt {retry_count} failed: {error_msg}, retrying in {retry_delay:.1f} seconds...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"All backup attempts failed after {self.max_retries} retries: {error_msg}")
                
                if not backup_success:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.critical(f"Stopping automatic backup after {consecutive_failures} consecutive failures")
                        self.stop_flag.set()
                        break
                
                # Clean up old auto backups if needed
                self._cleanup_old_auto_backups()
                
            except Exception as e:
                logger.error(f"Error in periodic backup: {e}", exc_info=True)
                # Sleep a bit before continuing to next cycle
                time.sleep(60)
    
    def _cleanup_old_auto_backups(self):
        """Removes old automatic backups based on count and size limits"""
        try:
            # 获取所有备份
            backups = self.backup_service.list_backups()
            
            # 过滤自动备份
            auto_backups = [b for b in backups if 
                          "automatic backup" in b.get("description", "").lower()]
            
            # 按时间戳排序（最新的在前）
            auto_backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # 获取配置的大小限制（默认50GB）
            max_total_size_gb = float(self.config.get("MAX_AUTO_BACKUP_SIZE_GB", "50"))
            max_total_size_bytes = max_total_size_gb * 1024 * 1024 * 1024
            
            # 跟踪已使用的总大小
            total_size = 0
            backups_to_keep = []
            
            # 首先保留最新的必需备份数量
            min_backups_to_keep = max(1, min(self.max_auto_backups // 2, 3))  # 至少保留1个，最多保留3个最新备份
            backups_to_keep.extend(auto_backups[:min_backups_to_keep])
            total_size = sum(b.get("size_bytes", 0) for b in backups_to_keep)
            
            # 处理剩余的备份
            for backup in auto_backups[min_backups_to_keep:]:
                backup_size = backup.get("size_bytes", 0)
                
                # 如果添加此备份后仍在限制范围内，且未超过最大数量限制，则保留
                if (total_size + backup_size <= max_total_size_bytes and 
                    len(backups_to_keep) < self.max_auto_backups):
                    backups_to_keep.append(backup)
                    total_size += backup_size
                else:
                    # 删除不满足条件的备份
                    backup_id = backup.get("id")
                    if backup_id:
                        logger.info(f"Cleaning up old auto backup: {backup_id} (size: {backup_size/(1024*1024):.2f}MB)")
                        self.backup_service.delete_backup(backup_id)
            
            logger.info(f"Backup cleanup completed. Keeping {len(backups_to_keep)} backups, total size: {total_size/(1024*1024*1024):.2f}GB")
        
        except Exception as e:
            logger.error(f"Error cleaning up old auto backups: {e}", exc_info=True)