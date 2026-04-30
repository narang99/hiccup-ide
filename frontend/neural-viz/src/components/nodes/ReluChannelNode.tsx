interface ReluChannelNodeProps {
  channelIndex: number;
}

export const ReluChannelNodeData = ({ channelIndex }: ReluChannelNodeProps) => (
  <div className="text-center">
    <div className="font-bold text-xs">ReLU{channelIndex}</div>
    <div className="text-xs text-gray-500">Ch{channelIndex}</div>
  </div>
);