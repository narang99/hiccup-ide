import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createOrUpdateWorkGraph } from '../fetchers/graph';

export const PruneGraphButton = () => {
  const [isPruning, setIsPruning] = useState(false);
  const navigate = useNavigate();

  // Hardcoded values as specified in requirements
  const modelAlias = 'example-model';
  const inputAlias = 'first-input';
  const workflowName = 'default-workflow';
  const graphAlias = 'default_pruned_graph';

  const handlePruneGraph = async () => {
    setIsPruning(true);
    try {
      await createOrUpdateWorkGraph(modelAlias, inputAlias, workflowName, graphAlias);
      navigate('/prune-graph/');
    } catch (error) {
      console.error('Failed to prune graph:', error);
    } finally {
      setIsPruning(false);
    }
  };

  return (
    <button
      onClick={handlePruneGraph}
      disabled={isPruning}
      style={{
        padding: '8px 12px',
        background: isPruning ? 'rgba(255,255,255,0.1)' : 'rgba(147, 51, 234, 0.2)',
        border: isPruning ? '1px solid rgba(255,255,255,0.2)' : '1px solid rgba(147, 51, 234, 0.5)',
        borderRadius: 6,
        color: isPruning ? 'rgba(255,255,255,0.4)' : '#a855f7',
        fontSize: 11,
        fontWeight: 600,
        cursor: isPruning ? 'not-allowed' : 'pointer',
        transition: 'all 0.2s ease',
        width: '100%',
      }}
    >
      {isPruning ? 'Pruning...' : 'Prune Graph'}
    </button>
  );
};
