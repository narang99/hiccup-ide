import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchWorkspace, createWork, type ModelTree } from '../fetchers/workspace';

const LandingPage: React.FC = () => {
  const [workspace, setWorkspace] = useState<ModelTree[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadWorkspace = () => {
    setLoading(true);
    fetchWorkspace()
      .then(data => {
        setWorkspace(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchWorkspace()
      .then(data => {
        setWorkspace(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
  }, []);

  const handleCreateWork = async (modelAlias: string, inputAlias: string) => {
    const name = prompt("Enter a name for the new work:");
    if (!name) return;

    try {
      await createWork(modelAlias, inputAlias, name);
      loadWorkspace();
    } catch (err) {
      alert(`Failed to create work: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

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
          {error && <div style={{ color: 'red' }}>Error: {error}</div>}
          {!loading && !error && workspace.length === 0 && (
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
                          <Link 
                            to={`/models/${model.alias}/${input.alias}/${work.alias}`}
                            style={{ 
                              color: 'var(--text-h)', 
                              textDecoration: 'none',
                              padding: '2px 6px',
                              margin: '2px 0',
                              borderRadius: '4px',
                              display: 'inline-block',
                              transition: 'all 0.2s'
                            }}
                            className="explorer-item"
                          >
                            📄 {work.name} 
                            <span style={{ opacity: 0.5, fontSize: '0.85em', marginLeft: '8px' }}>
                              {work.alias}
                            </span>
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
