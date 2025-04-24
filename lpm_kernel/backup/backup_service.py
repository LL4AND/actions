import os
import shutil
from datetime import datetime
import uuid
from pathlib import Path
import json
from typing import Optional, Dict, List
from tqdm import tqdm
from ..common.logging import logger
from .backup_integrity import BackupIntegrity

# Placeholder for backup service implementation

class BackupService:
    def __init__(self, config):
        self.config = config
        # Initialize backup directory path from config, default to 'backups' relative to base_dir
        base_dir = Path(config.get("BASE_DIR", "."))
        self.backup_base_dir = base_dir / config.get("BACKUP_DIR", "backups")
        self.backup_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup directory initialized at: {self.backup_base_dir}")

        # Initialize backup integrity checker
        encryption_key = config.get("BACKUP_ENCRYPTION_KEY")
        self.integrity_checker = BackupIntegrity(encryption_key)

        # Define source directories/files to be backed up, relative to base_dir
        self.source_paths = [
            Path(config.get("RESOURCES_DIR", "resources")),
            Path(config.get("DB_FILE", "data/sqlite/lpm.db")).parent, # Backup the whole sqlite dir
            Path(config.get("CHROMA_PERSIST_DIRECTORY", "data/chroma_db"))
        ]
        # Convert relative paths to absolute paths based on base_dir
        self.absolute_source_paths = [base_dir / p for p in self.source_paths]
        
        # Backup settings
        self.compress_backup = config.get("BACKUP_COMPRESS", "true").lower() == "true"
        self.encrypt_backup = config.get("BACKUP_ENCRYPT", "false").lower() == "true"

    def create_backup(self, description=None, tags: Optional[List[str]] = None, name: Optional[str] = None):
        """Creates a new backup."""
        backup_id = str(uuid.uuid4())
        timestamp = datetime.now()
        backup_folder_name = timestamp.strftime("%Y%m%d_%H%M%S_") + backup_id[:8]
        backup_path = self.backup_base_dir / backup_folder_name

        try:
            backup_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating backup '{backup_id}' in {backup_path}")

            copied_items = []
            total_size = 0

            # Copy source paths to backup directory
            for src_path in self.absolute_source_paths:
                if not src_path.exists():
                    logger.warning(f"Source path {src_path} does not exist, skipping.")
                    continue
                
                dest_path = backup_path / src_path.name
                try:
                    if src_path.is_dir():
                        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                        logger.info(f"Copied directory {src_path} to {dest_path}")
                    elif src_path.is_file():
                        shutil.copy2(src_path, dest_path)
                        logger.info(f"Copied file {src_path} to {dest_path}")
                    else:
                        logger.warning(f"Source path {src_path} is neither a file nor a directory, skipping.")
                        continue
                    
                    # Calculate size (basic implementation, might need refinement for large dirs)
                    current_size = sum(f.stat().st_size for f in dest_path.glob('**/*') if f.is_file())
                    total_size += current_size
                    copied_items.append(str(src_path.relative_to(Path(self.config.get("BASE_DIR", ".")))))

                except Exception as copy_e:
                    logger.error(f"Error copying {src_path} to {dest_path}: {copy_e}", exc_info=True)
                    # Decide if we should continue or fail the whole backup
                    # For now, let's log and continue

            # Process backup files (compress and encrypt if enabled)
            processed_items = []
            with tqdm(total=len(copied_items), desc="Processing backup files") as pbar:
                for item in copied_items:
                    item_path = backup_path / Path(item).name
                    if self.compress_backup:
                        item_path = self.integrity_checker.compress_file(item_path)
                    if self.encrypt_backup:
                        item_path = self.integrity_checker.encrypt_file(item_path)
                    processed_items.append(str(item_path.relative_to(backup_path)))
                    pbar.update(1)

            # Generate integrity manifest
            integrity_manifest = self.integrity_checker.generate_integrity_manifest(backup_path)

            # Create metadata file
            metadata = {
                "id": backup_id,
                "name": name or backup_folder_name,
                "timestamp": timestamp.isoformat(),
                "description": description or "Manual backup",
                "tags": tags or [],
                "size_bytes": total_size,
                "items": copied_items,
                "processed_items": processed_items,
                "compression_enabled": self.compress_backup,
                "encryption_enabled": self.encrypt_backup,
                "integrity_manifest": integrity_manifest
            }
            metadata_file = backup_path / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Backup '{backup_id}' created successfully.")
            return metadata

        except Exception as e:
            logger.error(f"Failed to create backup '{backup_id}': {e}", exc_info=True)
            # Clean up partially created backup directory if needed
            if backup_path.exists():
                try:
                    shutil.rmtree(backup_path)
                    logger.info(f"Cleaned up failed backup directory: {backup_path}")
                except Exception as cleanup_e:
                    logger.error(f"Error cleaning up failed backup directory {backup_path}: {cleanup_e}")
            return None # Indicate failure

    def list_backups(self):
        """Lists available backups by reading metadata files."""
        backups = []
        if not self.backup_base_dir.exists():
            logger.warning(f"Backup directory {self.backup_base_dir} does not exist.")
            return backups

        for backup_dir in self.backup_base_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "backup_metadata.json"
                if metadata_file.exists() and metadata_file.is_file():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            # Basic validation: check if 'id' and 'timestamp' exist
                            if 'id' in metadata and 'timestamp' in metadata:
                                backups.append(metadata)
                            else:
                                logger.warning(f"Invalid metadata file found in {backup_dir}, missing required fields.")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode metadata file: {metadata_file}")
                    except Exception as e:
                        logger.error(f"Error reading metadata file {metadata_file}: {e}", exc_info=True)
                else:
                    logger.debug(f"Skipping directory {backup_dir}, no metadata file found.")
        
        # Sort backups by timestamp, newest first
        backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        logger.info(f"Found {len(backups)} backups.")
        return backups

    def restore_backup(self, backup_id, verify_integrity: bool = True):
        """Restores data from a specific backup."""
        logger.info(f"Attempting to restore backup: {backup_id}")
        # Find the backup directory
        backup_to_restore = None
        for backup_dir in self.backup_base_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.endswith(backup_id[:8]): # Simple check based on folder name convention
                metadata_file = backup_dir / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if metadata.get('id') == backup_id:
                                backup_to_restore = backup_dir
                                break
                    except Exception as e:
                        logger.error(f"Error reading metadata for potential restore candidate {backup_dir}: {e}")
        
        if not backup_to_restore:
            logger.error(f"Backup with ID {backup_id} not found.")
            return {"status": "error", "message": f"Backup {backup_id} not found."}

        logger.info(f"Found backup directory to restore: {backup_to_restore}")
        
        try:
            # Get metadata to know what was backed up
            metadata_file = backup_to_restore / "backup_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Verify backup integrity if requested
            if verify_integrity:
                logger.info("Verifying backup integrity...")
                if not self.integrity_checker.verify_backup_integrity(backup_to_restore, metadata.get('integrity_manifest', {})):
                    return {"status": "error", "message": "Backup integrity verification failed"}
            
            # Get base directory from config
            base_dir = Path(self.config.get("BASE_DIR", "."))
            
            # Restore each backed up item
            restored_items = []
            for item in metadata.get('items', []):
                source_path = backup_to_restore / Path(item).name
                target_path = base_dir / item
                
                if not source_path.exists():
                    logger.warning(f"Source path {source_path} does not exist in backup, skipping.")
                    continue
                
                try:
                    # Create parent directory if it doesn't exist
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Remove existing data if it exists
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                            logger.info(f"Removed existing directory: {target_path}")
                        else:
                            target_path.unlink()
                            logger.info(f"Removed existing file: {target_path}")
                    
                    # Copy from backup to target
                    if source_path.is_dir():
                        shutil.copytree(source_path, target_path)
                        logger.info(f"Restored directory from {source_path} to {target_path}")
                    else:
                        shutil.copy2(source_path, target_path)
                        logger.info(f"Restored file from {source_path} to {target_path}")
                    
                    restored_items.append(item)
                except Exception as e:
                    logger.error(f"Error restoring {item}: {e}", exc_info=True)
            
            # Return success with details
            return {
                "status": "success", 
                "message": f"Successfully restored backup {backup_id}",
                "restored_items": restored_items,
                "timestamp": metadata.get('timestamp'),
                "description": metadata.get('description')
            }
            
        except Exception as e:
            logger.error(f"Error during restore process: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to restore backup {backup_id}: {e}"}

    def delete_backup(self, backup_id):
        """Deletes a specific backup directory."""
        logger.info(f"Attempting to delete backup: {backup_id}")
        backup_to_delete = None
        for backup_dir in self.backup_base_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.endswith(backup_id[:8]): # Simple check
                metadata_file = backup_dir / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if metadata.get('id') == backup_id:
                                backup_to_delete = backup_dir
                                break
                    except Exception as e:
                        logger.error(f"Error reading metadata for potential delete candidate {backup_dir}: {e}")

        if backup_to_delete:
            try:
                shutil.rmtree(backup_to_delete)
                logger.info(f"Successfully deleted backup directory: {backup_to_delete}")
                return {"status": "success", "message": f"Backup {backup_id} deleted."}
            except Exception as e:
                logger.error(f"Error deleting backup directory {backup_to_delete}: {e}", exc_info=True)
                return {"status": "error", "message": f"Failed to delete backup {backup_id}: {e}"}
        else:
            logger.warning(f"Backup with ID {backup_id} not found for deletion.")
            return {"status": "error", "message": f"Backup {backup_id} not found."}