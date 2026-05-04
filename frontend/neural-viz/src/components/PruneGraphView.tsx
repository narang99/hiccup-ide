import { useEffect, useState, useCallback } from 'react';
import { useNodesState, useEdgesState, Panel } from '@xyflow/react';
import { getPruningStatus, saveWorkSaliencyMaps, startPruning, finalizePruning, type PruningStatusResponse } from '../fetchers/graph';
import SharedCanvas from './SharedCanvas';
import { useFetcherType } from '../hooks/useFetcherType';
import { useSingleLayer } from '../hooks/useSingleLayer';
import { DataTypeSelector } from './SharedCanvas/Controls/DataTypeSelector';
import { ColormapSelector } from './SharedCanvas/Controls/ColormapSelector';
import { useGlobalStateControl } from '../hooks/useGlobalStateControl';
import type { SelectedNode } from '../types/node';
import { PruneHistogramPreview } from './prune_preview/PruneHistogramPreview';
import type { ActivationFilterAlgorithm } from '../types/activationFiltering';

export default function PruneGraphView() {
  const [status, setStatus] = useState<PruningStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);

  // Hardcoded values as specified in requirements
  const modelAlias = 'example-model';
  const inputAlias = 'first-input';
  const workflowName = 'default-workflow';
  const graphAlias = 'default_pruned_graph';
  const pageDirection = 'TB';

  const { fetcherType, setWorkGraph } = useFetcherType();

  const fetchStatus = useCallback(async () => {
    try {
      const data = await getPruningStatus(modelAlias, inputAlias, workflowName, graphAlias);
      setStatus(data);
      
      // Update global context so fetchers know to look in the scratchpad if a session is active
      if (data.session_active) {
        setWorkGraph(workflowName, graphAlias);
      } else {
        setWorkGraph(null, null);
      }
    } catch (err) {
      console.error('Failed to fetch pruning status:', err);
      setStatusError('Failed to load pruning progress.');
    } finally {
      setStatusLoading(false);
    }
  }, [modelAlias, inputAlias, workflowName, graphAlias, setWorkGraph]);

  useEffect(() => {
    const load = async () => {
      await fetchStatus();
    };
    load();
  }, [fetchStatus]);

  const handleStartPruning = async () => {
    setIsSaving(true);
    try {
      await startPruning(modelAlias, inputAlias, workflowName, graphAlias);
      await fetchStatus();
    } catch (err) {
      console.error('Failed to start pruning:', err);
      alert('Failed to start pruning session.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleFinalizePruning = async () => {
    setIsFinalizing(true);
    try {
      await finalizePruning(modelAlias, inputAlias, workflowName, graphAlias);
      await fetchStatus();
    } catch (err) {
      console.error('Failed to finalize pruning:', err);
      alert('Failed to finalize pruning session.');
    } finally {
      setIsFinalizing(false);
    }
  };

  // Find the first layer that is not done
  const firstIncompleteLayer = status?.layers.total.find(
    (layer) => !status.layers.done.includes(layer)
  );

  const {
    nodes: initialNodes,
    edges: initialEdges,
    loading: layerLoading,
    error: layerError,
    modelNode
  } = useSingleLayer(modelAlias, firstIncompleteLayer || '', fetcherType, pageDirection);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { scalingMode } = useGlobalStateControl({ nodes, fetcherType, setNodes, modelAlias, inputAlias });

  // Sync state when hook returns new nodes (e.g. after data fetch)
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges, scalingMode]);

  const maybeParentNode = nodes.find(n => n.type === "LayerNode");
  const parentNode = (maybeParentNode === undefined) ? null : maybeParentNode as unknown as SelectedNode;

  const handleSaveAndNext = async () => {
    if (!parentNode) return;
    
    setIsSaving(true);
    const items = nodes
      .filter(n => n.type === "ActivationFlowNode" && n.parentId === parentNode.id)
      .map(n => ({
        coordinate: n.data.coordinate as string,
        algorithm: (n.data.filterAlgorithm as ActivationFilterAlgorithm) || { type: 'Id' }
      }));

    try {
      await saveWorkSaliencyMaps(modelAlias, inputAlias, workflowName, graphAlias, items);
      
      // Refresh status to move to next layer
      await fetchStatus();
    } catch (err) {
      console.error('Failed to save and next:', err);
      alert('Failed to save progress.');
    } finally {
      setIsSaving(false);
    }
  };

  if (statusLoading) {
    return <div className="flex items-center justify-center h-screen">Loading pruning status...</div>;
  }

  if (statusError) {
    return <div className="flex items-center justify-center h-screen text-red-500">{statusError}</div>;
  }

  if (!status) {
    return <div className="flex items-center justify-center h-screen">No status available.</div>;
  }

  // Show start screen if no session is active in the temporary space
  if (!status.session_active) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <h1 className="text-2xl font-bold mb-4">Pruning Session</h1>
        <p className="mb-8 text-gray-400">Start a new session to prune the graph layer by layer in a temporary workspace.</p>
        <button
          onClick={handleStartPruning}
          disabled={isSaving}
          style={{
            padding: '12px 24px',
            background: '#a855f7',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          {isSaving ? 'Starting...' : 'Start Pruning Session'}
        </button>
      </div>
    );
  }

  if (!firstIncompleteLayer) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <h1 className="text-2xl font-bold text-green-500">Pruning Complete!</h1>
        <p className="mt-4 mb-8 text-gray-400">All layers have been processed in the temporary space.</p>
        <button
            onClick={handleFinalizePruning}
            disabled={isFinalizing}
            style={{
                padding: '12px 24px',
                background: '#22c55e',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontWeight: 600,
                cursor: 'pointer',
            }}
        >
            {isFinalizing ? 'Finalizing...' : 'Finalize and Commit Changes'}
        </button>
        <button
            onClick={handleStartPruning}
            className="mt-4 text-sm text-gray-500 hover:text-white"
            style={{ background: 'none', border: 'none', cursor: 'pointer' }}
        >
            Restart Session (Discard Temp)
        </button>
      </div>
    );
  }

  if (layerLoading) {
    return <div className="flex items-center justify-center h-screen">Loading layer data...</div>;
  }

  if (layerError) {
    return <div className="flex items-center justify-center h-screen text-red-500">Error: {layerError}</div>;
  }

  if (!modelNode) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <div className="text-xl font-bold text-red-500">Layer "{firstIncompleteLayer}" not found in model "{modelAlias}"</div>
      </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
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
          <PruneHistogramPreview selectedNode={parentNode} />
          
          <button
            onClick={handleSaveAndNext}
            disabled={isSaving}
            style={{
              padding: '10px 20px',
              background: isSaving ? 'rgba(168, 85, 247, 0.5)' : '#a855f7',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 600,
              cursor: isSaving ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 14px 0 rgba(168, 85, 247, 0.39)',
              transition: 'all 0.2s ease',
              marginTop: '10px'
            }}
          >
            {isSaving ? 'Saving...' : 'Save and Next'}
          </button>

          <button
            onClick={handleStartPruning}
            style={{
              background: 'none',
              border: 'none',
              color: 'rgba(255,255,255,0.4)',
              fontSize: '11px',
              cursor: 'pointer',
              textDecoration: 'underline'
            }}
          >
            Restart Session
          </button>
        </Panel>

        {/* Status Overlay */}
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          padding: '12px 16px',
          background: 'rgba(13, 13, 20, 0.88)',
          border: '1px solid rgba(255,255,255,0.09)',
          borderRadius: 10,
          backdropFilter: 'blur(10px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          zIndex: 1000,
          color: '#fff',
        }}>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
            Pruning Progress (Session)
          </div>
          <div style={{ fontSize: '14px', fontWeight: 500 }}>
            Processing: <span style={{ color: '#a855f7', fontWeight: 700 }}>{firstIncompleteLayer}</span>
          </div>
          <div style={{ fontSize: '11px', marginTop: '4px', color: 'rgba(255,255,255,0.6)' }}>
            {status.layers.done.length} / {status.layers.total.length} layers complete
          </div>
        </div>
      </SharedCanvas>
    </div>
  );
}
