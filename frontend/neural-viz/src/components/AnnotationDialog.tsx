interface AnnotationDialogProps {
    isOpen: boolean;
    gridCoord: [number, number] | null;
    onClose: () => void;
}

export default function AnnotationDialog({ isOpen, gridCoord, onClose }: AnnotationDialogProps) {
    if (!isOpen) return null;

    return (
        <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            backdropFilter: 'blur(4px)',
        }}>
            <div style={{
                background: '#1e1e2e',
                padding: '24px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.1)',
                width: '320px',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
            }}>
                <h3 style={{ margin: '0 0 16px 0', color: '#fff', fontSize: '16px' }}>
                    Pixel Annotation {gridCoord ? `(${gridCoord[0]}, ${gridCoord[1]})` : ''}
                </h3>
                
                <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '4px' }}>Label</label>
                    <input 
                        type="text" 
                        style={{
                            width: '100%',
                            boxSizing: 'border-box',
                            background: '#0d0d14',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '6px',
                            padding: '8px 12px',
                            color: '#fff',
                            fontSize: '14px',
                            outline: 'none',
                        }}
                        placeholder="Enter label..."
                        autoFocus
                    />
                </div>

                <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '4px' }}>Note</label>
                    <textarea 
                        style={{
                            width: '100%',
                            boxSizing: 'border-box',
                            background: '#0d0d14',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '6px',
                            padding: '8px 12px',
                            color: '#fff',
                            fontSize: '14px',
                            outline: 'none',
                            minHeight: '80px',
                            resize: 'vertical',
                        }}
                        placeholder="Enter note..."
                    />
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
                    <button 
                        onClick={onClose}
                        style={{
                            padding: '8px 16px',
                            background: 'transparent',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '6px',
                            color: 'rgba(255,255,255,0.7)',
                            cursor: 'pointer',
                            fontSize: '13px',
                        }}
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={onClose}
                        style={{
                            padding: '8px 16px',
                            background: '#3b82f6',
                            border: 'none',
                            borderRadius: '6px',
                            color: '#fff',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: 600,
                        }}
                    >
                        Save
                    </button>
                </div>
            </div>
        </div>
    );
}
