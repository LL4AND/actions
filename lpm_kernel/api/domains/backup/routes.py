from flask import Blueprint, request, jsonify, current_app
from lpm_kernel.backup.backup_service import BackupService
from lpm_kernel.configs.config import Config # Import Config 

backup_bp = Blueprint('backup', __name__, url_prefix='/api/backups')

# Instantiate BackupService using the application config
# Note: This assumes Config is a singleton or can be instantiated here.
# A better approach might involve dependency injection or accessing config via current_app.
config = Config.from_env() # Attempt to load config
backup_service = BackupService(config=config) # Pass the loaded config

@backup_bp.route('/', methods=['POST'])
def create_backup_route():
    """API endpoint to manually trigger a backup."""
    try:
        data = request.get_json() or {}
        description = data.get('description')
        result = backup_service.create_backup(description=description)
        if result:
            return jsonify(result), 201
        else:
            return jsonify({"status": "error", "message": "Failed to create backup"}), 500
    except Exception as e:
        current_app.logger.error(f"Backup creation failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Backup creation failed: {str(e)}"}), 500

@backup_bp.route('/', methods=['GET'])
def list_backups_route():
    """API endpoint to list available backups."""
    try:
        backups = backup_service.list_backups()
        return jsonify({"status": "success", "backups": backups}), 200
    except Exception as e:
        current_app.logger.error(f"Listing backups failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Listing backups failed: {str(e)}"}), 500

@backup_bp.route('/<string:backup_id>/restore', methods=['POST'])
def restore_backup_route(backup_id):
    """API endpoint to restore data from a specific backup."""
    try:
        result = backup_service.restore_backup(backup_id)
        if result.get('status') == 'error':
            return jsonify(result), 404 if 'not found' in result.get('message', '').lower() else 500
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Restore failed: {str(e)}"}), 500

@backup_bp.route('/<string:backup_id>', methods=['DELETE'])
def delete_backup_route(backup_id):
    """API endpoint to delete a specific backup."""
    try:
        result = backup_service.delete_backup(backup_id)
        if result.get('status') == 'error':
            return jsonify(result), 404 if 'not found' in result.get('message', '').lower() else 500
        return jsonify(result), 200
    except Exception as e:
        current_app.logger.error(f"Backup deletion failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Backup deletion failed: {str(e)}"}), 500

# TODO: Add error handling
# TODO: Add more robust error handling (e.g., specific error codes)
# TODO: Consider dependency injection for BackupService