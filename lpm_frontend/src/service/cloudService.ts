import { Request } from '../utils/request';
import type { CommonResponse } from '../types/responseModal';
import type { TrainProgress as LocalTrainProgress } from './train'; // Assuming this is the correct path and type for local progress
import type { AxiosPromise } from 'axios';

export interface CloudModel {
  model_id: string;
  model_name: string;
  description?: string;
  created_at?: string;
}

export const listAvailableModels = (): AxiosPromise<CommonResponse<CloudModel[]>> => {
  return Request<CommonResponse<CloudModel[]>>({
    method: 'get',
    url: '/api/cloud_service/list_available_models'
  });
};

export interface TrainingJobInfo {
  job_id: string;
  timestamp: string;
  status: string;
  model_id?: string;
  message?: string;
}

// Re-using LocalTrainProgress structure for cloud progress data as per user's example and request.
export type CloudProgressData = LocalTrainProgress;

export interface CloudTrainingProgressResponseData {
  job_id: string | null;
  progress: CloudProgressData;
  timestamp: string;
}

export const getCloudTrainingProgress = (): AxiosPromise<
  CommonResponse<CloudTrainingProgressResponseData>
> => {
  return Request<CommonResponse<CloudTrainingProgressResponseData>>({
    method: 'get',
    url: '/api/cloud_service/train/progress'
  });
};

// Stop cloud training
export const stopCloudTraining = (): AxiosPromise<CommonResponse<unknown>> => {
  return Request<CommonResponse<unknown>>({
    method: 'post',
    url: '/api/cloud_service/train/stop'
  });
};

// Reset cloud training progress
export const resetCloudTrainingProgress = (): AxiosPromise<CommonResponse<unknown>> => {
  return Request<CommonResponse<unknown>>({
    method: 'post',
    url: '/api/cloud_service/train/progress/reset'
  });
};

// Resume cloud training
export const resumeCloudTraining = (): AxiosPromise<CommonResponse<unknown>> => {
  return Request<CommonResponse<unknown>>({
    method: 'post',
    url: '/api/cloud_service/train/resume'
  });
};

export interface CloudTrainingStatus {
  status: string; // e.g., "PROCESSING", "SUCCEEDED", "FAILED"
  model_id?: string; // Available when SUCCEEDED
  message?: string;
  details?: Record<string, unknown>; // For more detailed status information
}

export const startCloudTraining = (params: {
  base_model: string;
  training_type?: string;
  data_synthesis_mode?: string;
  hyper_parameters?: Record<string, unknown>;
}): AxiosPromise<CommonResponse<unknown>> => {
  return Request<CommonResponse<unknown>>({
    method: 'post',
    url: '/api/cloud_service/train/start',
    data: params
  });
};

export const getCloudTrainingStatus = (
  jobId: string
): AxiosPromise<CommonResponse<CloudTrainingStatus>> => {
  return Request<CommonResponse<CloudTrainingStatus>>({
    method: 'get',
    url: `/api/cloud_service/train/status/training/${jobId}`
  });
};

export interface CloudDeployment {
  base_model: string;
  deployed_model: string;
  name: string;
  status: string; // e.g., "RUNNING", "PENDING", "STOPPED"
}

export interface DeploymentListResponse {
  count: number;
  deployments: CloudDeployment[];
}

export const listDeployments = (): AxiosPromise<CommonResponse<DeploymentListResponse>> => {
  return Request<CommonResponse<DeploymentListResponse>>({
    method: 'get',
    url: '/api/cloud_service/train/list_deployments'
  });
};

// Cloud inference request interface
export interface CloudInferenceRequest {
  messages: { role: 'user' | 'assistant' | 'system'; content: string }[];
  model_id: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  // Knowledge retrieval parameters for hybrid architecture
  enable_l0_retrieval?: boolean;
  enable_l1_retrieval?: boolean;
  role_id?: string;
}

// Cloud inference function using fetch for streaming support
export const runCloudInference = async (
  request: CloudInferenceRequest,
  signal?: AbortSignal
): Promise<Response> => {
  const response = await fetch('/api/cloud_service/train/inference', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    },
    body: JSON.stringify(request),
    signal
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response;
};

// Cloud service control interfaces
export interface CloudServiceStartRequest {
  model_id: string;
  model_name?: string;
}

export interface CloudServiceResponse {
  service_type: string;
  model_id?: string;
  model_name?: string;
  status: string;
  model_data?: {
    model_id: string;
    model_name: string;
    model_path: string;
    service_endpoint: string;
    base_model?: string;
  };
}

// Start cloud service
export const startCloudService = (
  request: CloudServiceStartRequest
): AxiosPromise<CommonResponse<CloudServiceResponse>> => {
  return Request<CommonResponse<CloudServiceResponse>>({
    method: 'post',
    url: '/api/cloud_service/service/start',
    data: request
  });
};

// Stop cloud service
export const stopCloudService = (): AxiosPromise<CommonResponse<CloudServiceResponse>> => {
  return Request<CommonResponse<CloudServiceResponse>>({
    method: 'post',
    url: '/api/cloud_service/service/stop'
  });
};

// Get cloud service status
export const getCloudServiceStatus = (): AxiosPromise<CommonResponse<CloudServiceResponse>> => {
  return Request<CommonResponse<CloudServiceResponse>>({
    method: 'get',
    url: '/api/cloud_service/service/status'
  });
};
