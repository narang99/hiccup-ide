import { usePruned } from '../hooks/usePruned';

export const PrunedGraphToggle = () => {
  const { isPruned, setPruned } = usePruned();

  const handleToggle = () => {
    setPruned(!isPruned, 'default_pruned_graph');
  };

  return (
    <button
      onClick={handleToggle}
      style={{
        padding: '8px 12px',
        background: isPruned ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.1)',
        border: isPruned ? '1px solid rgba(34, 197, 94, 0.5)' : '1px solid rgba(239, 68, 68, 0.3)',
        borderRadius: 6,
        color: isPruned ? '#4ade80' : '#ef4444',
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
        background: isPruned ? '#4ade80' : '#ef4444',
        boxShadow: isPruned ? '0 0 8px rgba(74, 222, 128, 0.5)' : 'none'
      }} />
      {isPruned ? 'Pruned View: ON' : 'Pruned View: OFF'}
    </button>
  );
};
