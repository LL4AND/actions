from flask import Flask, request
from .common.repository.database_session import DatabaseSession, Base
from .common.logging import logger
from .api import init_routes
from .api.file_server.handler import FileServerHandler
import os
import atexit
import subprocess


def ensure_cuda_support():
    """Run the script to ensure CUDA support for GPU acceleration"""
    if os.path.exists('/app/docker/app/ensure_cuda_support.sh'):
        try:
            logger.info("Checking CUDA support for GPU acceleration...")
            result = subprocess.run(
                ['/app/docker/app/ensure_cuda_support.sh'], 
                check=True, 
                capture_output=True, 
                text=True
            )
            logger.info(f"CUDA check result: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run CUDA support check: {str(e)}")
            logger.error(f"Error output: {e.stderr}")
    else:
        logger.info("CUDA support script not found, skipping check.")


def create_app():
    app = Flask(__name__)

    # Ensure CUDA support for GPU acceleration
    if os.getenv('IN_DOCKER_ENV') == '1':
        ensure_cuda_support()

    # Initialize database connection
    try:
        DatabaseSession.initialize()
        logger.info("Database connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {str(e)}")
        raise

        # Add CORS support

    @app.after_request
    def after_request(response):
        # Allow all origins in development environment
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type,Authorization"
        )
        response.headers.add(
            "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
        )
        return response

    # Create file server handler
    file_handler = FileServerHandler(
        os.path.join(os.getenv("APP_ROOT", "/app"), "resources", "raw_content")
    )

    @app.route("/raw_content/", defaults={"path": ""})
    @app.route("/raw_content/<path:path>")
    def serve_content(path=""):
        return file_handler.handle_request(path, request.path)

    # Register all routes
    init_routes(app)

    # Clean up database connection only when the application shuts down
    @app.teardown_appcontext
    def cleanup_db(exception):
        pass

    return app


app = create_app()


@atexit.register
def cleanup():
    DatabaseSession.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
