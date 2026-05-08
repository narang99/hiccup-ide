import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export interface PinnedWorkflowLink {
  url: string;
  name: string;
  alias: string;
}

interface PinnedWorkflowsButtonProps {
  links: PinnedWorkflowLink[];
  loading?: boolean;
}

const PinnedWorkflowsButton: React.FC<PinnedWorkflowsButtonProps> = ({
  links,
  loading = false
}) => {
  const [isHovered, setIsHovered] = useState(false);

  if (links.length === 0 && !loading) {
    return null; // Don't show button if no pinned works
  }

  return (
    <div 
      style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <button
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '14px 14px',
          background: 'rgba(13, 13, 20, 0.88)',
          border: '1px solid rgba(255,255,255,0.09)',
          borderRadius: 10,
          backdropFilter: 'blur(10px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          color: 'rgba(255,255,255,0.75)',
          fontSize: 11,
          fontWeight: 600,
          cursor: 'pointer',
          minWidth: 'fit-content',
        }}
      >
        📌 Pinned ({links.length})
      </button>

      {isHovered && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: '0',
            background: 'rgba(13, 13, 20, 0.95)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 10,
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            zIndex: 1000,
            minWidth: '300px',
            maxWidth: '400px',
            padding: '8px 0',
            marginTop: '4px'
          }}
        >
          {loading ? (
            <div style={{ padding: '12px 16px', color: 'rgba(255,255,255,0.5)' }}>
              Loading pinned workflows...
            </div>
          ) : links.length === 0 ? (
            <div style={{ padding: '12px 16px', color: 'rgba(255,255,255,0.5)' }}>
              No pinned workflows for this model
            </div>
          ) : (
            links.map((link) => (
              <Link
                key={link.alias}
                to={link.url}
                target="_blank"
                style={{
                  display: 'block',
                  width: '90%',
                  padding: '10px 16px',
                  textDecoration: 'none',
                  color: 'rgba(255,255,255,0.75)',
                  transition: 'background-color 0.2s',
                  fontSize: 11,
                  fontWeight: 600,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: '2px', fontSize: 11 }}>
                  {link.name}
                </div>
                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.5)' }}>
                  {link.alias}
                </div>
              </Link>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default PinnedWorkflowsButton;