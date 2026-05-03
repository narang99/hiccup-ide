import { useFetcherType } from '../hooks/useFetcherType';

export const PrunedGraphToggle = () => {
  const { workAlias, setWorkGraph } = useFetcherType();

  // Hardcoded values as specified in requirements
  const workflowName = 'default-workflow';
  const graphAlias = 'default_pruned_graph';

  const isPruned = workAlias === workflowName;

  const handleToggle = () => {
    if (isPruned) {
      setWorkGraph(null, null);
    } else {
      setWorkGraph(workflowName, graphAlias);
    }
  };

  return (
    <button
      onClick={handleToggle}
      style={{
        padding: '8px 12px',
        background: isPruned ? 'rgba(34, 197, 94, 0.2)' : 'rgba(255, 255, 255, 0.05)',
        border: isPruned ? '1px solid rgba(34, 197, 94, 0.5)' : '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 6,
        color: isPruned ? '#4ade80' : 'rgba(255, 255, 255, 0.6)',
        fontSize: 11,
        fontWeight: 600,
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '6px'
      }}
    >
      <div style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: isPruned ? '#4ade80' : 'rgba(255, 255, 255, 0.2)',
        boxShadow: isPruned ? '0 0 8px rgba(74, 222, 128, 0.5)' : 'none'
      }} />
      {isPruned ? 'Pruned View: ON' : 'Pruned View: OFF'}
    </button>
  );
};
