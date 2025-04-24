import threading
import time
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
        while not self.stop_flag.is_set():
            try:
                # Sleep for the specified interval
                for _ in range(self.backup_interval * 60):
                    if self.stop_flag.is_set():
                        return
                    time.sleep(1)
                
                # Create backup with retries
                description = f"Training in-progress automatic backup{' for ' + model_name if model_name else ''}"
                logger.info(f"Creating periodic backup: {description}")
                
                retry_count = 0
                while retry_count < self.max_retries:
                    try:
                        backup_result = self.backup_service.create_backup(
                            description=description,
                            tags=["auto", "training", model_name] if model_name else ["auto", "training"]
                        )
                        if backup_result:
                            logger.info("Backup created successfully")
                            break
                        raise Exception("Backup creation failed")
                    except Exception as backup_error:
                        retry_count += 1
                        if retry_count < self.max_retries:
                            logger.warning(f"Backup attempt {retry_count} failed, retrying in {self.retry_delay} seconds...")
                            time.sleep(self.retry_delay)
                        else:
                            logger.error(f"All backup attempts failed after {self.max_retries} retries")
                            raise backup_error
                
                # Clean up old auto backups if needed
                self._cleanup_old_auto_backups()
                
            except Exception as e:
                logger.error(f"Error in periodic backup: {e}", exc_info=True)
                # Sleep a bit before continuing to next cycle
                time.sleep(60)
    
    def _cleanup_old_auto_backups(self):
        """Removes old automatic backups to save space"""
        try:
            # Get all backups
            backups = self.backup_service.list_backups()
            
            # Filter automatic backups
            auto_backups = [b for b in backups if 
                          "automatic backup" in b.get("description", "").lower()]
            
            # Sort by timestamp (newest first)
            auto_backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Delete old backups beyond the maximum limit
            if len(auto_backups) > self.max_auto_backups:
                for backup in auto_backups[self.max_auto_backups:]:
                    backup_id = backup.get("id")
                    if backup_id:
                        logger.info(f"Cleaning up old auto backup: {backup_id}")
                        self.backup_service.delete_backup(backup_id)
        
        except Exception as e:
            logger.error(f"Error cleaning up old auto backups: {e}", exc_info=True)