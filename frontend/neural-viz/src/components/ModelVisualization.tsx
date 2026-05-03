import { useModelVisualization } from '../hooks/useModelVisualization';
import SharedCanvas from './SharedCanvas';
import { useFetcherType } from '../hooks/useFetcherType';
import { Panel } from '@xyflow/react';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { PruneGraphButton } from './PruneGraphButton';
import { AttachedToSelectedNodeLayerSettings } from './prune_preview/AttachedToSelectedNodeTopKSumSliderPreview';
import { useCallback } from 'react';
import { loadWorkflowThresholds } from '../fetchers/threshold';

export default function ModelVisualization() {
  const { fetcherType } = useFetcherType();
  const pageDirection = "TB";
  const {
    modelData,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
  } = useModelVisualization(fetcherType, pageDirection);

  // Hardcoded values moved from LayerSettings
  const modelAlias = 'example-model';
  const inputAlias = 'first-input';
  const workflowName = 'default-workflow';

  const onLoadInitialThresholds = useCallback(() => {
    return loadWorkflowThresholds(modelAlias, inputAlias, workflowName);
  }, []);

  if (!modelData) {
    return <div className="flex items-center justify-center h-screen">Loading model...</div>;
  }

  return (
    <SharedCanvas
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      pageDirection={pageDirection}
      fitView
    >
        <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
          <DataTypeSelector />
          <ColormapSelector />
          <AttachedToSelectedNodeLayerSettings 
            onLoadInitialThresholds={onLoadInitialThresholds}
          />
          <PruneGraphButton />
        </Panel>
    </SharedCanvas>
  );
}