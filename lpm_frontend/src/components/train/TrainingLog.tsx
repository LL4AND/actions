import { useEffect, useRef, useState } from 'react';

interface TrainingLogProps {
  trainingDetails: {
    message: string;
    timestamp: string;
  }[];
}

const TrainingLog: React.FC<TrainingLogProps> = ({ trainingDetails }: TrainingLogProps) => {
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const userScrollTimeout = useRef<NodeJS.Timeout | null>(null);
  const [isAutoScrollEnabled, setIsAutoScrollEnabled] = useState(true);

  // Smooth scroll console to bottom
  const smoothScrollConsole = () => {
    if (consoleEndRef.current && isAutoScrollEnabled && !isUserScrolling) {
      const consoleContainer = consoleEndRef.current.closest('.overflow-y-auto');

      if (consoleContainer instanceof HTMLElement) {
        consoleContainer.scrollTo({
          top: consoleContainer.scrollHeight,
          behavior: 'smooth'
        });
      }
    }
  };

  useEffect(() => {
    // Set up scroll event listener to detect user scrolling
    const handleUserScroll = () => {
      if (!consoleEndRef.current) return;
      
      const consoleContainer = consoleEndRef.current.closest('.overflow-y-auto');
      
      if (!(consoleContainer instanceof HTMLElement)) return;
      
      // Check if scrolled away from bottom
      const isScrolledToBottom = 
        Math.abs((consoleContainer.scrollHeight - consoleContainer.scrollTop) - consoleContainer.clientHeight) < 50;
      
      // If scrolled away from bottom, consider it manual scrolling
      if (!isScrolledToBottom) {
        setIsUserScrolling(true);

        // Clear any existing timeout
        if (userScrollTimeout.current) {
          clearTimeout(userScrollTimeout.current);
        }

        // Reset the flag after a delay
        userScrollTimeout.current = setTimeout(() => {
          setIsUserScrolling(false);
        }, 5000); // 5 seconds delay before allowing auto-scroll again
      } else {
        // If at bottom, not considered manual scrolling
        setIsUserScrolling(false);
        if (userScrollTimeout.current) {
          clearTimeout(userScrollTimeout.current);
          userScrollTimeout.current = null;
        }
      }
    };

    // Find the console container and attach the scroll listener
    if (consoleEndRef.current) {
      const consoleContainer = consoleEndRef.current.closest('.overflow-y-auto');

      if (consoleContainer instanceof HTMLElement) {
        consoleContainer.addEventListener('scroll', handleUserScroll);

        // Cleanup function
        return () => {
          consoleContainer.removeEventListener('scroll', handleUserScroll);

          if (userScrollTimeout.current) {
            clearTimeout(userScrollTimeout.current);
          }
        };
      }
    }
  }, []);

  useEffect(() => {
    if (trainingDetails.length > 0) {
      smoothScrollConsole();
    }
  }, [trainingDetails, isAutoScrollEnabled]);

  const toggleAutoScroll = () => {
    setIsAutoScrollEnabled(!isAutoScrollEnabled);
    if (!isAutoScrollEnabled) {
      // If we're re-enabling auto-scroll, scroll to bottom immediately
      setIsUserScrolling(false);
      setTimeout(smoothScrollConsole, 50);
    }
  };

  return (
    <div className="mt-4">
      <div className="flex justify-between items-center mb-2">
        <h4 className="text-sm font-medium text-gray-700">Training Log</h4>
        <div className="flex items-center">
          <button 
            onClick={toggleAutoScroll}
            className={`text-xs px-2 py-1 rounded ${
              isAutoScrollEnabled 
                ? 'bg-blue-500 text-white hover:bg-blue-600' 
                : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
            } transition-colors`}
          >
            {isAutoScrollEnabled ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
          </button>
        </div>
      </div>
      <div className="bg-gray-900 rounded-lg p-4 h-[600px] overflow-y-auto font-mono text-xs">
        <div className="space-y-1">
          {trainingDetails.length > 0 ? (
            trainingDetails.map((detail, index) => (
              <div key={detail.timestamp + detail.message + index} className="text-gray-300">
                {detail.message}
              </div>
            ))
          ) : (
            <div className="text-gray-300">
              No training logs available. Start training to see logs here.
            </div>
          )}
          <div ref={consoleEndRef} />
        </div>
      </div>
    </div>
  );
};

export default TrainingLog;
