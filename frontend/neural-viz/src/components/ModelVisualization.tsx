import { useModelVisualization } from '../hooks/useModelVisualization';
import SharedCanvas from './SharedCanvas';
import { useFetcherType } from '../hooks/useFetcherType';
import { Panel } from '@xyflow/react';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { LayerSettings } from './LayerSettings';
import { PruneGraphButton } from './PruneGraphButton';

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
          <LayerSettings />
          <PruneGraphButton />
        </Panel>
    </SharedCanvas>
  );
}