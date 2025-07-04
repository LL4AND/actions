import { useState, useRef } from 'react';
import { runCloudInference, type CloudInferenceRequest } from '../service/cloudService';

export interface ChatRequest {
  messages: ChatHistory[];
  metadata?: ChatMetadata;
  temperature: number;
  max_tokens?: number;
  stream: boolean;
}

interface ChatMetadata {
  role_id?: string;
  enable_l0_retrieval: boolean;
  enable_l1_retrieval: boolean;
}
interface ChatHistory {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export const useSSE = (): {
  stopSSE: () => void;
  sendStreamMessage: (
    request: ChatRequest,
    isCloudModel?: boolean,
    cloudModelId?: string
  ) => Promise<void>;
  streaming: boolean;
  error: string | null;
  streamContent: string;
  streamRawContent: string;
  firstContentLoading: boolean;
} => {
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamContent, setStreamContent] = useState('');
  const [streamRawContent, setStreamRawContent] = useState('');
  const [firstContentLoading, setFirstContentLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const streamContentRef = useRef('');
  const streamRawContentRef = useRef('');

  const stopSSE = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    setStreaming(false);
  };

  const sendStreamMessage = async (
    request: ChatRequest,
    isCloudModel = false,
    cloudModelId?: string
  ) => {
    setStreaming(true);
    setError(null);
    setStreamContent('');
    setStreamRawContent('');
    setFirstContentLoading(true);
    streamContentRef.current = ''; // Clear this as well
    streamRawContentRef.current = ''; // Clear this as well

    // Use AbortController to cancel the request
    const controller = new AbortController();

    abortControllerRef.current = controller; // Store the controller in the ref

    const signal = controller.signal;

    try {
      let response: Response;

      if (isCloudModel && cloudModelId) {
        // Use cloud inference endpoint
        const cloudRequest: CloudInferenceRequest = {
          messages: request.messages.map((msg) => ({
            role: msg.role,
            content: msg.content
          })),
          model_id: cloudModelId,
          temperature: request.temperature,
          max_tokens: request.max_tokens || 2000,
          stream: true,
          // Pass knowledge retrieval parameters from metadata
          enable_l0_retrieval: request.metadata?.enable_l0_retrieval || false,
          enable_l1_retrieval: request.metadata?.enable_l1_retrieval || false,
          role_id: request.metadata?.role_id
        };

        response = await runCloudInference(cloudRequest, signal);
      } else {
        response = await fetch('/api/kernel2/chat', {
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
      }

      const reader = response.body?.getReader();

      if (!reader) {
        throw new Error('ReadableStream not supported');
      }

      const decoder = new TextDecoder();

      let interval: NodeJS.Timeout | null = null;

      const featchStream = async () => {
        try {
          const { done, value } = await reader.read();

          if (done) {
            if (interval) clearInterval(interval);

            setStreaming(false);

            return;
          }

          const chunk = decoder.decode(value);

          const lines = chunk.split('\n');

          for (const line of lines) {
            streamRawContentRef.current += line + '\n';
            setStreamRawContent(streamRawContentRef.current);

            if (!line.startsWith('data: ')) continue;

            if (line === 'data: [DONE]') break;

            const jsonChunk = line.slice(6).trim();

            if (!jsonChunk) continue;

            const parsedData = JSON.parse(jsonChunk);
            
            // Both cloud and local APIs now use the same format: {"choices": [{"delta": {"content": "..."}}]}
            const content = parsedData?.choices?.[0]?.delta?.content || '';

            if (content) {
              setFirstContentLoading(false);
              streamContentRef.current += content;
              setStreamContent(streamContentRef.current);
            }
          }
        } catch {
          setStreaming(false);

          if (interval) clearInterval(interval);
        }
      };

      interval = setInterval(featchStream, 10);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';

      setError(errorMessage);
      setStreamContent(errorMessage);
      console.error('Streaming error:', err);
      setStreaming(false);
    }
  };

  return {
    stopSSE,
    sendStreamMessage,
    streaming,
    error,
    streamContent,
    streamRawContent,
    firstContentLoading
  };
};
