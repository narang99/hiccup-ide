import { useEffect, useState } from 'react';
import SingleLayerView from './SingleLayerView';
import { getPruningStatus, type PruningStatusResponse } from '../fetchers/graph';

export default function PruneGraphView() {
  const [status, setStatus] = useState<PruningStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Hardcoded values as specified in requirements
  const modelAlias = 'example-model';
  const inputAlias = 'first-input';
  const workflowName = 'default-workflow';
  const graphAlias = 'default_pruned_graph';
  const pageDirection = 'TB';

  useEffect(() => {
    async function fetchStatus() {
      try {
        const data = await getPruningStatus(modelAlias, inputAlias, workflowName, graphAlias);
        setStatus(data);
      } catch (err) {
        console.error('Failed to fetch pruning status:', err);
        setError('Failed to load pruning progress.');
      } finally {
        setLoading(false);
      }
    }

    fetchStatus();
  }, [modelAlias, inputAlias, workflowName, graphAlias]);

  if (loading) {
    return <div className="flex items-center justify-center h-screen">Loading pruning status...</div>;
  }

  if (error) {
    return <div className="flex items-center justify-center h-screen text-red-500">{error}</div>;
  }

  if (!status) {
    return <div className="flex items-center justify-center h-screen">No status available.</div>;
  }

  // Find the first layer that is not done
  const firstIncompleteLayer = status.layers.total.find(
    (layer) => !status.layers.done.includes(layer)
  );

  if (!firstIncompleteLayer) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <h1 className="text-2xl font-bold text-green-500">Pruning Complete!</h1>
        <p className="mt-4 text-gray-400">All layers have been processed.</p>
      </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <SingleLayerView
        modelAlias={modelAlias}
        inputAlias={inputAlias}
        layerId={firstIncompleteLayer}
        pageDirection={pageDirection}
      />
      {/* Status Overlay */}
      {/* <div style={{
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
          Pruning Progress
        </div>
        <div style={{ fontSize: '14px', fontWeight: 500 }}>
          Processing: <span style={{ color: '#a855f7', fontWeight: 700 }}>{firstIncompleteLayer}</span>
        </div>
        <div style={{ fontSize: '11px', marginTop: '4px', color: 'rgba(255,255,255,0.6)' }}>
          {status.layers.done.length} / {status.layers.total.length} layers complete
        </div>
      </div> */}
    </div>
  );
}
