import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchWorkspace, createWork, pinWork, unpinWork } from '../fetchers/workspace';

const LandingPage: React.FC = () => {
  const queryClient = useQueryClient();

  const { data: workspace = [], isLoading: loading, error } = useQuery({
    queryKey: ['workspace'],
    queryFn: fetchWorkspace,
  });

  const createWorkMutation = useMutation({
    mutationFn: ({ modelAlias, inputAlias, name }: { modelAlias: string, inputAlias: string, name: string }) => 
      createWork(modelAlias, inputAlias, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workspace'] });
    },
    onError: (err) => {
      alert(`Failed to create work: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  const pinToggleMutation = useMutation({
    mutationFn: async ({ modelAlias, inputAlias, workAlias, isPinned }: { modelAlias: string, inputAlias: string, workAlias: string, isPinned: boolean }) => {
      if (isPinned) {
        return unpinWork(modelAlias, inputAlias, workAlias);
      } else {
        return pinWork(modelAlias, inputAlias, workAlias);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workspace'] });
    },
    onError: (err, variables) => {
      alert(`Failed to ${variables.isPinned ? 'unpin' : 'pin'} work: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  const handleCreateWork = (modelAlias: string, inputAlias: string) => {
    const name = prompt("Enter a name for the new work:");
    if (!name) return;
    createWorkMutation.mutate({ modelAlias, inputAlias, name });
  };

  const handlePinToggle = (modelAlias: string, inputAlias: string, workAlias: string, isPinned: boolean) => {
    pinToggleMutation.mutate({ modelAlias, inputAlias, workAlias, isPinned });
  };

  const errorMessage = error instanceof Error ? error.message : (error ? String(error) : null);

  return (
    <div style={{ 
      padding: '60px 20px', 
      maxWidth: '900px', 
      margin: '0 auto', 
      height: '100vh', 
      overflowY: 'auto',
      boxSizing: 'border-box'
    }}>
      <header style={{ marginBottom: '48px' }}>
        <h1 style={{ margin: '0 0 16px 0', fontSize: '48px' }}>Hiccup IDE</h1>
        <p style={{ fontSize: '20px', opacity: 0.8 }}>
          Neural Network Visualization & Pruning Environment
        </p>
      </header>
      
      <section>
        <h2 style={{ marginBottom: '24px', borderBottom: '1px solid var(--border)', paddingBottom: '12px' }}>
          Workspace Explorer
        </h2>
        
        <div style={{ 
          fontFamily: 'var(--mono)', 
          fontSize: '15px', 
          lineHeight: '1.8',
          backgroundColor: 'var(--code-bg)',
          padding: '24px',
          borderRadius: '8px',
          border: '1px solid var(--border)'
        }}>
          {loading && <div>Loading workspace...</div>}
          {errorMessage && <div style={{ color: 'red' }}>Error: {errorMessage}</div>}
          {!loading && !errorMessage && workspace.length === 0 && (
            <div>No models found. Register a new model to get started.</div>
          )}

          {workspace.map((model, mIdx) => (
            <div key={model.alias} style={{ marginBottom: '24px' }}>
              <div style={{ fontWeight: 'bold', color: 'var(--accent)', fontSize: '16px' }}>
                {mIdx === workspace.length - 1 ? '└──' : '├──'} 📁 {model.name} 
                <span style={{ opacity: 0.5, fontSize: '0.85em', fontWeight: 'normal', marginLeft: '8px' }}>
                  {model.alias}/
                </span>
              </div>
              
              {model.inputs.map((input, iIdx) => {
                const isLastModel = mIdx === workspace.length - 1;
                const isLastInput = iIdx === model.inputs.length - 1;
                const prefix = isLastModel ? '    ' : '│   ';
                
                return (
                  <div key={input.alias}>
                    <div style={{ color: 'var(--text-h)' }}>
                      {prefix}{isLastInput ? '└──' : '├──'} 📥 {input.name}
                      <span style={{ opacity: 0.5, fontSize: '0.85em', marginLeft: '8px' }}>
                        {input.alias}/
                      </span>
                    </div>
                    
                    {input.works.map((work, wIdx) => {
                      const isLastWork = wIdx === input.works.length - 1;
                      const subPrefix = prefix + (isLastInput ? '    ' : '│   ');
                      
                      return (
                        <div key={work.alias} style={{ display: 'flex', alignItems: 'center' }}>
                          <span style={{ color: 'var(--text)', whiteSpace: 'pre' }}>
                            {subPrefix}{isLastWork ? '└──' : '├──'} 
                          </span>
                          <button
                            onClick={() => handlePinToggle(model.alias, input.alias, work.alias, work.is_pinned)}
                            style={{
                              background: 'none',
                              border: 'none',
                              color: work.is_pinned ? '#ffd700' : '#666',
                              cursor: 'pointer',
                              fontSize: '14px',
                              padding: '2px 4px',
                              marginRight: '4px'
                            }}
                            title={work.is_pinned ? 'Unpin work' : 'Pin work'}
                          >
                            {work.is_pinned ? '📌' : '📎'}
                          </button>
                          <Link 
                            to={`/models/${model.alias}/${input.alias}/${work.alias}`}
                            style={{ 
                              color: 'var(--text-h)', 
                              textDecoration: 'none',
                              padding: '2px 6px',
                              margin: '2px 0',
                              borderRadius: '4px',
                              display: 'inline-block',
                              transition: 'all 0.2s',
                              borderLeft: work.is_pinned ? '2px solid #ffd700' : 'none',
                              paddingLeft: work.is_pinned ? '8px' : '6px'
                            }}
                            className="explorer-item"
                          >
                            📄 {work.name} 
                            <span style={{ opacity: 0.5, fontSize: '0.85em', marginLeft: '8px' }}>
                              {work.alias}
                            </span>
                            {work.is_pinned && <span style={{ color: '#ffd700', fontSize: '0.8em', marginLeft: '6px' }}>★</span>}
                          </Link>
                        </div>
                      );
                    })}
                    
                    {/* Add Work action */}
                    <div style={{ display: 'flex', alignItems: 'center', opacity: 0.6 }}>
                      <span style={{ color: 'var(--text)', whiteSpace: 'pre' }}>
                        {prefix + (isLastInput ? '    ' : '│   ')}└── 
                      </span>
                      <button 
                        onClick={() => handleCreateWork(model.alias, input.alias)}
                        style={{ 
                          background: 'none',
                          border: 'none',
                          color: 'var(--accent)',
                          cursor: 'pointer',
                          fontFamily: 'inherit',
                          fontSize: 'inherit',
                          padding: '2px 6px',
                          textDecoration: 'underline'
                        }}
                      >
                        + New Work
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
          
          <div style={{ marginTop: '32px', borderTop: '1px dashed var(--border)', paddingTop: '16px' }}>
             <button 
              onClick={() => alert('Register new model')}
              style={{ 
                background: 'var(--accent)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '8px 16px',
                cursor: 'pointer',
                fontFamily: 'var(--sans)',
                fontWeight: 'bold'
              }}
            >
              + Register New Model
            </button>
          </div>
        </div>
      </section>

      <style>{`
        .explorer-item:hover {
          background-color: var(--accent-bg) !important;
          color: var(--accent) !important;
          transform: translateX(4px);
        }
      `}</style>
    </div>
  );
};

export default LandingPage;
