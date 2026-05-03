import { useNodesState, useEdgesState, Panel } from '@xyflow/react';
import { useEffect } from 'react';
import SharedCanvas from './SharedCanvas';
import { useFetcherType } from '../hooks/useFetcherType';
import { useSingleLayer } from '../hooks/useSingleLayer';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { useGlobalStateControl } from '../hooks/useGlobalStateControl';

interface SingleLayerViewProps {
  modelAlias: string;
  inputAlias: string;
  layerId: string;
  pageDirection?: "TB" | "LR";
}

export default function SingleLayerView({ 
  modelAlias, 
  inputAlias,
  layerId, 
  pageDirection = "TB" 
}: SingleLayerViewProps) {
  const { fetcherType } = useFetcherType();

  const {
    nodes: initialNodes,
    edges: initialEdges,
    loading,
    error,
    modelNode
  } = useSingleLayer(modelAlias, layerId, fetcherType, pageDirection);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { scalingMode } = useGlobalStateControl({nodes, fetcherType, setNodes, modelAlias, inputAlias})

  // Sync state when hook returns new nodes (e.g. after data fetch)
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges, scalingMode]);

  if (loading) {
    return <div className="flex items-center justify-center h-screen">Loading layer data...</div>;
  }

  if (error) {
    return <div className="flex items-center justify-center h-screen text-red-500">Error: {error}</div>;
  }

  if (!modelNode) {
    return (
        <div className="flex flex-col items-center justify-center h-screen">
            <div className="text-xl font-bold text-red-500">Layer "{layerId}" not found in model "{modelAlias}"</div>
        </div>
    );
  }


  return (
    <div style={{ width: '100%', height: '100%' }}>
        <SharedCanvas
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            pageDirection={pageDirection}
            fitView
        />
          <Panel position="top-right" style={{ display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'flex-end' }}>
            <DataTypeSelector />
            <ColormapSelector />
          </Panel>
    </div>
  );
}
