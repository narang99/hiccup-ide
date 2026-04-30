import { type NodeFetchers } from '../../fetchers';
import { ActivationDisplay } from '../ActivationDisplay';

interface ReluChannelNodeProps {
  channelIndex: number;
  layerId: string;
  fetchers?: NodeFetchers;
}

export const ReluChannelNodeData = ({ channelIndex, layerId, fetchers }: ReluChannelNodeProps) => {
  const coordinate = `${layerId}.out_${channelIndex}`;
  
  return (
    <div className="text-center">
      <div className="font-bold text-xs">ReLU{channelIndex}</div>
      
      {/* Activation Display */}
      {fetchers?.activation ? (
        <ActivationDisplay 
          coordinate={coordinate}
          fetcher={fetchers.activation}
          maxSize={30}
          className="my-1"
        />
      ) : (
        <div className="w-8 h-8 bg-gray-400 rounded border my-1" />
      )}
      
      <div className="text-xs text-gray-500">Ch{channelIndex}</div>
    </div>
  );
};