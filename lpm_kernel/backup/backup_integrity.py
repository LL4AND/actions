import hashlib
import zlib
import json
from pathlib import Path
from typing import Dict, List, Optional
from cryptography.fernet import Fernet
from ..common.logging import logger

class BackupIntegrity:
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize backup integrity checker with optional encryption."""
        self.encryption_key = encryption_key.encode() if encryption_key else Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def compress_file(self, file_path: Path, compressed_path: Optional[Path] = None, compression_level: int = 6) -> Path:
        """Compress a file using zlib with configurable compression level (1-9)."""
        if not compressed_path:
            compressed_path = file_path.with_suffix('.gz')
        
        try:
            # 使用分块读取以处理大文件
            chunk_size = 1024 * 1024  # 1MB chunks
            compressor = zlib.compressobj(level=compression_level)
            with open(file_path, 'rb') as f_in, open(compressed_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    compressed_chunk = compressor.compress(chunk)
                    if compressed_chunk:
                        f_out.write(compressed_chunk)
                # 确保写入所有剩余的压缩数据
                f_out.write(compressor.flush())
            
            # 计算压缩比
            original_size = file_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100
            logger.info(f"Compressed {file_path.name}: {compression_ratio:.1f}% reduction (Level {compression_level})")
            
            return compressed_path
        except Exception as e:
            logger.error(f"Error compressing {file_path}: {e}")
            if compressed_path.exists():
                compressed_path.unlink()  # 清理失败的压缩文件
            raise
    
    def decompress_file(self, compressed_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decompress a zlib compressed file."""
        if not output_path:
            output_path = compressed_path.with_suffix('')
        
        try:
            with open(compressed_path, 'rb') as f_in:
                compressed_data = f_in.read()
            data = zlib.decompress(compressed_data)
            with open(output_path, 'wb') as f_out:
                f_out.write(data)
            return output_path
        except Exception as e:
            logger.error(f"Error decompressing {compressed_path}: {e}")
            raise
    
    def encrypt_file(self, file_path: Path, encrypted_path: Optional[Path] = None) -> Path:
        """Encrypt a file using Fernet symmetric encryption."""
        if not encrypted_path:
            encrypted_path = file_path.with_suffix('.enc')
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            encrypted_data = self.fernet.encrypt(data)
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            return encrypted_path
        except Exception as e:
            logger.error(f"Error encrypting {file_path}: {e}")
            raise
    
    def decrypt_file(self, encrypted_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decrypt a Fernet encrypted file."""
        if not output_path:
            output_path = encrypted_path.with_suffix('')
        
        try:
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            return output_path
        except Exception as e:
            logger.error(f"Error decrypting {encrypted_path}: {e}")
            raise
    
    def generate_integrity_manifest(self, backup_path: Path) -> Dict:
        """Generate integrity manifest for all files in backup."""
        manifest = {
            'files': [],
            'total_files': 0,
            'total_size': 0
        }
        
        for file_path in backup_path.rglob('*'):
            if file_path.is_file():
                file_info = {
                    'path': str(file_path.relative_to(backup_path)),
                    'size': file_path.stat().st_size,
                    'checksum': self.calculate_checksum(file_path)
                }
                manifest['files'].append(file_info)
                manifest['total_files'] += 1
                manifest['total_size'] += file_info['size']
        
        return manifest
    
    def verify_backup_integrity(self, backup_path: Path, manifest: Dict) -> bool:
        """Verify backup integrity against manifest."""
        try:
            for file_info in manifest['files']:
                file_path = backup_path / file_info['path']
                if not file_path.exists():
                    logger.error(f"Missing file: {file_path}")
                    return False
                
                current_checksum = self.calculate_checksum(file_path)
                if current_checksum != file_info['checksum']:
                    logger.error(f"Checksum mismatch for {file_path}")
                    return False
                
                current_size = file_path.stat().st_size
                if current_size != file_info['size']:
                    logger.error(f"Size mismatch for {file_path}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error verifying backup integrity: {e}")
            return False