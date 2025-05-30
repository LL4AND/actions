import { create } from 'zustand';
import {
  getServiceStatus,
  getTrainProgress,
  type TrainProgress,
  type ServiceStatusRes
} from '@/service/train';
import { getCloudServiceStatus } from '@/service/cloudService';
import type { CommonResponse } from '@/types/responseModal';

export type ModelStatus = 'seed_identity' | 'memory_upload' | 'training' | 'trained';

export enum Status {
  SEED_IDENTITY = 'seed_identity',
  MEMORY_UPLOAD = 'memory_upload',
  TRAINING = 'training',
  TRAINED = 'trained'
}

export const statusRankMap = {
  [Status.SEED_IDENTITY]: 0,
  [Status.MEMORY_UPLOAD]: 1,
  [Status.TRAINING]: 2,
  [Status.TRAINED]: 2
};

interface ModelState {
  status: ModelStatus;
  error: boolean;
  isTraining: boolean;
  serviceStarted: boolean;
  isServiceStarting: boolean;
  isServiceStopping: boolean;
  trainingProgress: TrainProgress;
  trainSuspended: boolean;
  setStatus: (status: ModelStatus) => void;
  setError: (error: boolean) => void;
  setIsTraining: (isTraining: boolean) => void;
  fetchServiceStatus: () => Promise<CommonResponse<ServiceStatusRes>>;
  setServiceStarting: (isStarting: boolean) => void;
  setServiceStopping: (isStopping: boolean) => void;
  setTrainingProgress: (progress: TrainProgress) => void;
  setTrainSuspended: (suspended: boolean) => void;
  checkTrainStatus: () => Promise<void>;
  resetTrainingState: () => void;
}

const defaultTrainingProgress: TrainProgress = {
  current_stage: 'downloading_the_base_model',
  overall_progress: 0,
  stages: [
    {
      current_step: null,
      name: 'Downloading the Base Model',
      progress: 0,
      status: 'pending',
      steps: [{ completed: false, name: 'Model Download', status: 'pending' }]
    },
    {
      current_step: null,
      name: 'Activating the Memory Matrix',
      progress: 0,
      status: 'pending',
      steps: [
        { completed: false, name: 'List Documents', status: 'pending' },
        { completed: false, name: 'Generate Document Embeddings', status: 'pending' },
        { completed: false, name: 'Process Chunks', status: 'pending' },
        { completed: false, name: 'Chunk Embedding', status: 'pending' }
      ]
    },
    {
      current_step: null,
      name: 'Synthesize Your Life Narrative',
      progress: 0.0,
      status: 'pending',
      steps: [
        { completed: false, name: 'Extract Dimensional Topics', status: 'pending' },
        { completed: false, name: 'Map Your Entity Network', status: 'pending' }
      ]
    },
    {
      current_step: null,
      name: 'Prepare Training Data for Deep Comprehension',
      progress: 0.0,
      status: 'pending',
      steps: [
        { completed: false, name: 'Decode Preference Patterns', status: 'pending' },
        { completed: false, name: 'Reinforce Identity', status: 'pending' },
        { completed: false, name: 'Augment Content Retention', status: 'pending' }
      ]
    },
    {
      current_step: null,
      name: 'Training to create Second Me',
      progress: 0.0,
      status: 'pending',
      steps: [
        { completed: false, name: 'Train', status: 'pending' },
        { completed: false, name: 'Merge Weights', status: 'pending' },
        { completed: false, name: 'Convert Model', status: 'pending' }
      ]
    }
  ],
  status: 'pending'
};

export const useTrainingStore = create<ModelState>((set, get) => ({
  status: 'seed_identity',
  isTraining: false,
  serviceStarted: false,
  isServiceStarting: false,
  isServiceStopping: false,
  error: false,
  trainingProgress: defaultTrainingProgress,
  trainSuspended: false,
  setStatus: (status) => {
    const preStatus = get().status;

    //Only trained and running can be interchanged.
    if (statusRankMap[status] < statusRankMap[preStatus]) {
      return;
    }

    set({ status });
  },
  fetchServiceStatus: () => {
    // Check both local and cloud service status
    return Promise.all([
      getServiceStatus().catch(() => ({ data: { code: -1, data: { is_running: false } } })),
      getCloudServiceStatus().catch(() => ({
        data: {
          code: -1,
          data: {
            status: 'stopped',
            service_type: '',
            model_data: null
          }
        }
      }))
    ]).then(([localRes, cloudRes]) => {
      let serviceStarted = false;
      
      // Check local service status (uses is_running field)
      if (localRes.data.code === 0 && localRes.data.data.is_running) {
        serviceStarted = true;
        // Clear cloud model when local service is active
        import('@/utils/cloudModelUtils').then(({ clearActiveCloudModel }) => {
          clearActiveCloudModel();
        });
      }
      
      // Check cloud service status (uses status field with 'active'/'stopped')
      if (cloudRes.data.code === 0 && cloudRes.data.data.status === 'active') {
        serviceStarted = true;
        
        // Set active cloud model when cloud service is active
        const modelData = cloudRes.data.data.model_data;

        if (modelData) {
          import('@/utils/cloudModelUtils').then(({ setActiveCloudModel }) => {
            setActiveCloudModel({
              name: modelData.model_name || 'Unknown Cloud Model',
              deployed_model: modelData.model_id || '',
              base_model: modelData.base_model || '',
              status: 'active'
            });
          });
        }
      } else {
        // Clear cloud model when cloud service is not active
        import('@/utils/cloudModelUtils').then(({ clearActiveCloudModel }) => {
          clearActiveCloudModel();
        });
      }
      
      set({ serviceStarted });
      
      // Return the local service response for backward compatibility
      return localRes;
    });
  },
  setError: (error) => set({ error }),
  setIsTraining: (isTraining) => set({ isTraining }),
  setServiceStarting: (isStarting) => set({ isServiceStarting: isStarting }),
  setServiceStopping: (isStopping) => set({ isServiceStopping: isStopping }),
  setTrainingProgress: (progress) => set({ trainingProgress: progress }),
  setTrainSuspended: (suspended) => set({ trainSuspended: suspended }),
  resetTrainingState: () => set({ trainingProgress: defaultTrainingProgress }),
  checkTrainStatus: async () => {
    const config = JSON.parse(localStorage.getItem('trainingParams') || '{}');

    set({ error: false });

    try {
      // 确定要使用的模型名称，优先使用当前活动环境的模型名称
      const modelName =
        config.model_name ||
        config.local_model_name ||
        config.cloud_model_name ||
        'Qwen2.5-0.5B-Instruct';
      
      const res = await getTrainProgress({
        model_name: modelName,
        local_model_name: config.local_model_name || modelName,
        cloud_model_name: config.cloud_model_name || ''
      });

      if (res.data.code === 0) {
        const data = res.data.data;
        const { overall_progress, status } = data;

        const newProgress = data;

        if (newProgress.status === 'failed') {
          set({ error: true });
        }

        set((state) => {
          const newState = {
            ...state,
            trainingProgress: newProgress
          };

          if (status === 'suspended' || status === 'failed') {
            newState.trainSuspended = true;
          }

          // If total progress is 100%, set status to trained
          if (overall_progress === 100) {
            newState.status = 'trained';
          }
          // If there's any progress but not complete, set status to training
          else if (overall_progress > 0) {
            newState.status = 'training';
          }

          return newState;
        });
      }
    } catch (error) {
      console.error('Error checking training status:', error);
      set({ error: true });
    }
  }
}));
