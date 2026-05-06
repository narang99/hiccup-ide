import { useState, useEffect } from 'react';
import { 
    getKernelLabels, 
    addKernelLabel, 
    removeKernelLabel
} from '../fetchers/kernel_labels';

interface KernelLabelsDialogProps {
    isOpen: boolean;
    weightCoordinate: string;
    onClose: () => void;
}

export default function KernelLabelsDialog({ 
    isOpen, 
    weightCoordinate, 
    onClose 
}: KernelLabelsDialogProps) {
    const [labels, setLabels] = useState<string[]>([]);
    const [newLabel, setNewLabel] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        if (!isOpen || !weightCoordinate) {
            return;
        }

        let cancelled = false;
        
        const loadLabels = async () => {
            setIsLoading(true);
            try {
                const response = await getKernelLabels(weightCoordinate);
                if (!cancelled) {
                    setLabels(response.labels);
                }
            } catch (err) {
                if (!cancelled) {
                    console.error("Failed to load kernel labels:", err);
                    setLabels(['spurious']);
                }
            } finally {
                if (!cancelled) {
                    setIsLoading(false);
                }
            }
        };

        loadLabels();
        
        return () => {
            cancelled = true;
        };
    }, [isOpen, weightCoordinate]);

    const handleAddLabel = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newLabel.trim() || labels.includes(newLabel.trim())) {
            return;
        }

        setIsSaving(true);
        try {
            const response = await addKernelLabel(weightCoordinate, newLabel.trim());
            setLabels(response.labels);
            setNewLabel('');
        } catch (error) {
            console.error("Failed to add label:", error);
        } finally {
            setIsSaving(false);
        }
    };

    const handleRemoveLabel = async (labelToRemove: string) => {
        if (labelToRemove === 'spurious') {
            return;
        }

        setIsSaving(true);
        try {
            const response = await removeKernelLabel(weightCoordinate, labelToRemove);
            setLabels(response.labels);
        } catch (error) {
            console.error("Failed to remove label:", error);
        } finally {
            setIsSaving(false);
        }
    };

    const handleClose = (e: React.MouseEvent) => {
        e.stopPropagation();
        onClose();
    };

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
                width: '400px',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
            }}>
                <h3 style={{ margin: '0 0 16px 0', color: '#fff', fontSize: '16px' }}>
                    Manage Labels for Kernel
                </h3>
                
                {isLoading ? (
                    <div style={{ color: 'rgba(255,255,255,0.7)', padding: '20px', textAlign: 'center' }}>
                        Loading labels...
                    </div>
                ) : (
                    <>
                        <div style={{ marginBottom: '20px' }}>
                            <h4 style={{ margin: '0 0 12px 0', color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
                                Current Labels:
                            </h4>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                {labels.map(label => (
                                    <div
                                        key={label}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '6px',
                                            background: label === 'spurious' ? '#374151' : '#3b82f6',
                                            color: '#fff',
                                            padding: '4px 8px',
                                            borderRadius: '16px',
                                            fontSize: '12px',
                                            fontWeight: 500,
                                        }}
                                    >
                                        {label}
                                        {label !== 'spurious' && (
                                            <button
                                                onClick={() => handleRemoveLabel(label)}
                                                disabled={isSaving}
                                                style={{
                                                    background: 'transparent',
                                                    border: 'none',
                                                    color: 'rgba(255,255,255,0.8)',
                                                    cursor: 'pointer',
                                                    fontSize: '14px',
                                                    padding: 0,
                                                    marginLeft: '2px',
                                                }}
                                            >
                                                ×
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <form onSubmit={handleAddLabel} style={{ marginBottom: '24px' }}>
                            <label style={{ 
                                display: 'block', 
                                color: 'rgba(255,255,255,0.6)', 
                                fontSize: '12px', 
                                marginBottom: '4px' 
                            }}>
                                Add New Label
                            </label>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <input 
                                    type="text" 
                                    value={newLabel}
                                    onChange={(e) => setNewLabel(e.target.value)}
                                    style={{
                                        flex: 1,
                                        background: '#0d0d14',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        borderRadius: '6px',
                                        padding: '8px 12px',
                                        color: '#fff',
                                        fontSize: '14px',
                                        outline: 'none',
                                    }}
                                    placeholder="Enter new label..."
                                    disabled={isSaving}
                                />
                                <button 
                                    type="submit"
                                    disabled={isSaving || !newLabel.trim()}
                                    style={{
                                        padding: '8px 16px',
                                        background: '#10b981',
                                        border: 'none',
                                        borderRadius: '6px',
                                        color: '#fff',
                                        cursor: 'pointer',
                                        fontSize: '13px',
                                        fontWeight: 600,
                                        opacity: (isSaving || !newLabel.trim()) ? 0.5 : 1,
                                    }}
                                >
                                    Add
                                </button>
                            </div>
                        </form>
                    </>
                )}

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button 
                        onClick={handleClose}
                        disabled={isSaving}
                        style={{
                            padding: '8px 16px',
                            background: '#3b82f6',
                            border: 'none',
                            borderRadius: '6px',
                            color: '#fff',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: 600,
                            opacity: isSaving ? 0.7 : 1,
                        }}
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
}