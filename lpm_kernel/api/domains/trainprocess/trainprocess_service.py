import os
import re
import time
import threading
import gc
import subprocess
import psutil
from enum import Enum
from typing import Dict, List, Optional
import json

from lpm_kernel.configs.config import Config
from lpm_kernel.L1.utils import save_true_topics
from lpm_kernel.L1.serializers import NotesStorage
from lpm_kernel.kernel.note_service import NoteService
from lpm_kernel.L2.l2_generator import L2Generator
from lpm_kernel.L2.utils import save_hf_model
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.common.script_executor import ScriptExecutor
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.api.domains.trainprocess.progress_enum import Status
from lpm_kernel.api.domains.trainprocess.train_progress import TrainProgress
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.progress_holder import TrainProgressHolder
from lpm_kernel.api.domains.kernel.routes import store_l1_data
from lpm_kernel.train.training_params_manager import TrainingParamsManager
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.backup.auto_backup import AutoBackupManager
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.kernel.l1.l1_manager import (
    extract_notes_from_documents,
    document_service,
    get_latest_status_bio,
    get_latest_global_bio,
    generate_l1_from_l0,
)
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.configs.logging import get_train_process_logger, TRAIN_LOG_FILE

logger = get_train_process_logger()


class TrainProcessService:
    """Training process service (singleton pattern)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, current_model_name: str = None):
        if current_model_name is None and not self._initialized:
            raise ValueError("current_model_name cannot be None when initializing")
            
        if not self._initialized:
            # Generate a unique progress file name based on model name
            self.progress = TrainProgressHolder(current_model_name)
            self.model_name = current_model_name  # Set model name directly
            self._initialized = True
            
            # Initialize stop flag
            self.is_stopped = False
            self.current_step = None
            
            # Initialize L2 data dictionary
            self.l2_data = {
                "notes": None,
                "basic_info": None,
                "data_output_base_dir": None,
                "topics_path": None,
                "entitys_path": None,
                "graph_path": None,
                "config_path": None
            }
            self.l2_data_prepared = False
        
        # Update model name and progress instance if model name changes
        if current_model_name is not None and current_model_name != self.model_name:
            self.model_name = current_model_name
            # Create new progress instance with updated progress file name
            self.progress = TrainProgressHolder(current_model_name)
    
    @classmethod
    def get_instance(cls, current_model_name: str = None):
        """Get the current instance of TrainProcessService
        
        Args:
            current_model_name: Optional model name to update the instance with
            
        Returns:
            TrainProcessService: The singleton instance
        """
        if cls._instance is None:
            if current_model_name is None:
                logger.warning("current_model_name must be provided when creating a new instance")
                return None
            return cls(current_model_name)
        
        if current_model_name is not None:
            # Update the existing instance with new model name
            cls._instance.model_name = current_model_name
            cls._instance.progress = TrainProgressHolder(current_model_name)
            
        return cls._instance

    def list_documents(self):
        """List all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.IN_PROGRESS)            
            # Directly call document service instead of API
            documents = document_service.list_documents()
            # Mark step as completed if we found documents
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.COMPLETED)
                
            return [doc.to_dict() for doc in documents]
        except Exception as e:
            logger.error(f"List documents failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.FAILED)
            return []

    def generate_document_embeddings(self) -> bool:
        """Process embeddings for all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.IN_PROGRESS)
            documents = self.list_documents() 
            for doc in documents:
                doc_id = doc.get("id")

                # Directly call document service instead of API
                embedding = document_service.process_document_embedding(doc_id)
                if embedding is None:
                    logger.error(
                        f"Generate document embeddings failed for doc_id: {doc_id}"
                    )
                    self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
                    return False
                self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.COMPLETED)
                logger.info(f"Successfully generated embedding for document {doc_id}") 
            return True
        except Exception as e:
            logger.error(f"Generate document embeddings failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
            return False

    def process_chunks(self) -> bool:
        """Process document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.IN_PROGRESS)
            config = Config.from_env()
            chunker = DocumentChunker(
                chunk_size=int(config.get("DOCUMENT_CHUNK_SIZE")),
                overlap=int(config.get("DOCUMENT_CHUNK_OVERLAP")),
            )
            documents = document_service.list_documents()
            processed, failed = 0, 0

            chunk_service = ChunkService()
            for doc in documents:
                try:
                    if not doc.raw_content:
                        logger.warning(f"Document {doc.id} has no content, skipping...")
                        failed += 1
                        continue

                    # Split into chunks and save
                    chunks = chunker.split(doc.raw_content)
                    for chunk in chunks:
                        chunk.document_id = doc.id
                        chunk_service.save_chunk(chunk)

                    processed += 1
                    logger.info(
                        f"Document {doc.id} processed: {len(chunks)} chunks created"
                    )
                except Exception as e:
                    logger.error(f"Failed to process document {doc.id}: {str(e)}")
                    failed += 1      
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Process chunks failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.FAILED)
            return False

    def chunk_embedding(self) -> bool:
        """Process embeddings for all document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.IN_PROGRESS)
            documents = self.list_documents()
            for doc in documents:
                doc_id = doc.get("id")
                try:
                    # Directly call document service to generate chunk embeddings
                    processed_chunks = document_service.generate_document_chunk_embeddings(doc_id)
                    if not processed_chunks:
                        logger.warning(f"No chunks to process for document: {doc_id}")
                        continue
                except Exception as e:
                    logger.error(
                        f"Generate chunk embeddings failed for doc_id: {doc_id}: {str(e)}"
                    )
                    self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.FAILED)
                    return False
            # All documents' chunks processed successfully
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Generate chunk embeddings failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.FAILED)
            return False

    def extract_dimensional_topics(self) -> bool:
        """Extract dimensional topics (L0)"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.IN_PROGRESS)
            logger.info("Starting dimensional topics extraction (L0)...")
            
            # Generate L0 - Call document_service to analyze all documents
            logger.info("Generating L0 data...")
            analyzed_docs = document_service.analyze_all_documents()
            logger.info(f"Successfully analyzed {len(analyzed_docs)} documents for L0")
            
            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.COMPLETED)
            logger.info("Dimensional topics extraction (L0) completed successfully")
            return True

        except Exception as e:
            logger.error(f"Extract dimensional topics (L0) failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.FAILED)
            return False
            
    def generate_biography(self) -> bool:
        """Generate biography using L1 data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.IN_PROGRESS)
            logger.info("Starting biography generation...")

            # Generate L1 data and biography
            logger.info("Generating L1 data and biography...")
            l1_data = generate_l1_from_l0()
            logger.info("Successfully generated L1 data and biography")

            # Store L1 data
            with DatabaseSession.session() as session:
                store_l1_data(session, l1_data)

            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.COMPLETED)
            logger.info("Biography generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Biography generation failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.FAILED)
            return False

    def model_download(self) -> bool:
        """Download model"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.IN_PROGRESS)
            # Directly call save_hf_model function to download model
            logger.info(f"Starting model download: {self.model_name}")
            
            # Start monitoring the download progress in a separate thread
            monitor_thread = threading.Thread(target=self._monitor_model_download)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Start the actual download
            model_path = save_hf_model(self.model_name)
            
            if model_path and os.path.exists(model_path):
                logger.info(f"Model downloaded successfully to {model_path}")
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.COMPLETED)
                return True
            else:
                logger.error(f"Model path does not exist after download: {model_path}")
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
                return False

        except Exception as e:
            logger.error(f"Download model failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
            return False

    def map_your_entity_network(self)->bool:
        """Map entity network using notes and basic info"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.IN_PROGRESS)
            logger.info("Starting entity network mapping...")
        
            # Get or prepare L2 data
            self._prepare_l2_data()

            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources")
            )
            l2_generator.data_preprocess(self.l2_data["notes"], self.l2_data["basic_info"])
            
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.COMPLETED)
            logger.info("Entity network mapping completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Map entity network failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.FAILED)
            self._cleanup_resources()
            return False

    def decode_preference_patterns(self)->bool:
        """Decode preference patterns using notes and related data"""
        try:
            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            concurrency_threads = training_params.get("concurrency_threads")
            data_synthesis_mode = training_params.get("data_synthesis_mode")
            os.environ["CONCURRENCY_THREADS"] = str(concurrency_threads)
            os.environ["DATA_SYNTHESIS_MODE"] = data_synthesis_mode
            
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.IN_PROGRESS)
            logger.info("Starting preference patterns decoding...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Use data from l2_data dictionary
            training_params = TrainingParamsManager.get_latest_training_params()
            L2Generator(is_cot=training_params.get("is_cot", False)).gen_preference_data(                
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.COMPLETED)
            logger.info("Preference patterns decoding completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Decode preference patterns failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.FAILED)
            return False

    def reinforce_identity(self)->bool:
        """Reinforce identity using notes and related data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.IN_PROGRESS)
            logger.info("Starting identity reinforcement...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Get training parameters
            training_params = TrainingParamsManager.get_latest_training_params()
            # Use data from l2_data dictionary
            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources"), is_cot=training_params.get("is_cot", False)
                )  
            l2_generator.gen_selfqa_data(
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.COMPLETED)
            logger.info("Identity reinforcement completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Reinforce identity failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.FAILED)
            return False
            
    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        logger.info("Cleaning up resources to prevent memory leaks")
        
        # Clean up large data structures in l2_data dictionary
        for key in self.l2_data:
            self.l2_data[key] = None
        
        self.l2_data_prepared = False
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after cleanup
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def augment_content_retention(self) -> bool:
        """Augment content retention using notes, basic info and graph data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.IN_PROGRESS)
            logger.info("Starting content retention augmentation...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Get training parameters
            training_params = TrainingParamsManager.get_latest_training_params()
            # Use data from l2_data dictionary
            l2_generator = L2Generator(data_path=os.path.join(os.getcwd(), "resources"), is_cot=training_params.get("is_cot", False))
            l2_generator.gen_diversity_data(
                self.l2_data["notes"],
                self.l2_data["basic_info"],
                self.l2_data["data_output_base_dir"],
                self.l2_data["topics_path"],
                self.l2_data["entitys_path"],
                self.l2_data["graph_path"],
                self.l2_data["config_path"]
            )
            l2_generator.merge_json_files(self.l2_data["data_output_base_dir"])
            
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.COMPLETED)
            logger.info("Content retention augmentation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Augment content retention failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.FAILED)
            return False

    def _prepare_l2_data(self):
        """Prepare L2 data for training"""
        if self.l2_data_prepared:
            logger.info("L2 data already prepared, skipping preparation")
            return
            
        try:
            logger.info("Preparing L2 data...")
            
            # Get notes from database
            note_service = NoteService()
            notes = note_service.get_all_notes()
            
            # Get basic info
            status_bio = get_latest_status_bio()
            global_bio = get_latest_global_bio()
            
            # Combine into basic info dictionary
            basic_info = {
                "status_bio": status_bio,
                "global_bio": global_bio
            }
            
            # Create output directory
            data_output_base_dir = os.path.join(os.getcwd(), "data", "l2_data")
            os.makedirs(data_output_base_dir, exist_ok=True)
            
            # Set paths for L2 data
            topics_path = os.path.join(data_output_base_dir, "topics.json")
            entitys_path = os.path.join(data_output_base_dir, "entitys.json")
            graph_path = os.path.join(data_output_base_dir, "graph.json")
            config_path = os.path.join(data_output_base_dir, "config.json")
            
            # Store in l2_data dictionary
            self.l2_data["notes"] = notes
            self.l2_data["basic_info"] = basic_info
            self.l2_data["data_output_base_dir"] = data_output_base_dir
            self.l2_data["topics_path"] = topics_path
            self.l2_data["entitys_path"] = entitys_path
            self.l2_data["graph_path"] = graph_path
            self.l2_data["config_path"] = config_path
            
            self.l2_data_prepared = True
            logger.info("L2 data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"L2 data preparation failed: {str(e)}")
            self._cleanup_resources()
            raise

    def _monitor_model_download(self):
        """Monitor model download progress"""
        try:
            logger.info("Starting model download monitoring...")
            
            # Check every 5 seconds
            while True:
                # Sleep first to give download time to start
                time.sleep(5)
                
                # Check if download is still in progress
                if self.progress.get_step_status(ProcessStep.MODEL_DOWNLOAD) != Status.IN_PROGRESS:
                    logger.info("Model download monitoring stopped - download completed or failed")
                    break
                    
                # Log current memory usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"Current memory usage during download: {memory_info.rss / 1024 / 1024:.2f} MB")
                
        except Exception as e:
            logger.error(f"Error in model download monitoring: {str(e)}")

    def check_training_condition(self) -> bool:
        """Check if training conditions are met"""
        try:
            # Check if documents exist
            documents = self.list_documents()
            if not documents:
                logger.error("No documents found, training cannot proceed")
                return False
                
            # Check if model name is valid
            if not self.model_name:
                logger.error("Model name is not set, training cannot proceed")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking training conditions: {str(e)}")
            return False
            
    def reset_progress(self):
        """Reset training progress"""
        try:
            logger.info("Resetting training progress...")
            self.progress.reset_progress()
            logger.info("Training progress reset successfully")
        except Exception as e:
            logger.error(f"Error resetting training progress: {str(e)}")

    def start_process(self):
        """Start the training process"""
        try:
            logger.info("Starting training process...")
            
            # Reset stop flag
            self.is_stopped = False
            
            # Reset progress
            self.reset_progress()
            
            # Check training conditions
            if not self.check_training_condition():
                logger.error("Training conditions not met, aborting process")
                return False
                
            # Start the process
            self.progress.mark_status(Status.IN_PROGRESS)
            
            # Execute each step in sequence
            steps = [
                ("List documents", self.list_documents),
                ("Generate document embeddings", self.generate_document_embeddings),
                ("Process chunks", self.process_chunks),
                ("Generate chunk embeddings", self.chunk_embedding),
                ("Extract dimensional topics", self.extract_dimensional_topics),
                ("Generate biography", self.generate_biography),
                ("Download model", self.model_download),
                ("Map entity network", self.map_your_entity_network),
                ("Decode preference patterns", self.decode_preference_patterns),
                ("Reinforce identity", self.reinforce_identity),
                ("Augment content retention", self.augment_content_retention),
            ]
            
            for step_name, step_func in steps:
                if self.is_stopped:
                    logger.info("Process stopped by user")
                    self.progress.mark_status(Status.STOPPED)
                    return False
                    
                logger.info(f"Executing step: {step_name}")
                self.current_step = step_name
                
                # Execute step
                success = step_func()
                
                if not success:
                    logger.error(f"Step '{step_name}' failed, stopping process")
                    self.progress.mark_status(Status.FAILED)
                    return False
                    
            # All steps completed successfully
            logger.info("All steps completed successfully")
            self.progress.mark_status(Status.COMPLETED)
            return True
            
        except Exception as e:
            logger.error(f"Error in training process: {str(e)}")
            self.progress.mark_status(Status.FAILED)
            return False
            
    def stop_process(self):
        """Stop the training process"""
        try:
            logger.info("Stopping training process...")
            self.is_stopped = True
            logger.info("Training process stop flag set")
            return True
        except Exception as e:
            logger.error(f"Error stopping training process: {str(e)}")
            return False