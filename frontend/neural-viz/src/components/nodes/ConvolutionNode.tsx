import { type ModelNode, type ExpandedState } from '../../types/model';

interface ConvolutionNodeProps {
  modelNode: ModelNode;
  expandedState: ExpandedState;
  toggleExpand: (nodeId: string) => void;
}

export const ConvolutionNodeData = ({ 
  modelNode, 
  expandedState, 
  toggleExpand 
}: ConvolutionNodeProps) => {
  const isExpandable = modelNode.type === 'Conv2d' && modelNode.params.in_channels && modelNode.params.out_channels;
  const isExpanded = expandedState[modelNode.id];

  return (
    <div className="text-center">
      <div className="flex items-center justify-center gap-2">
        <div className="font-bold text-sm">{modelNode.type}</div>
        {isExpandable && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleExpand(modelNode.id);
            }}
            className="text-xs bg-gray-200 hover:bg-gray-300 px-2 py-1 rounded"
          >
            {isExpanded ? '−' : '+'}
          </button>
        )}
      </div>
      <div className="text-xs text-gray-600">{modelNode.id}</div>
      {modelNode.shape.length > 0 && (
        <div className="text-xs text-gray-500">
          {modelNode.shape.join(' × ')}
        </div>
      )}
      {Object.keys(modelNode.params).length > 0 && (
        <div className="text-xs text-gray-500">
          {Object.entries(modelNode.params)
            .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(',') : value}`)
            .join(', ')}
        </div>
      )}
    </div>
  );
};