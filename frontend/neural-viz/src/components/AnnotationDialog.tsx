import { useState, useEffect } from 'react';
import { listPOIs, savePOI } from '../fetchers/poi';
import { getKernelLabels } from '../fetchers/kernel_labels';
import { useAliases } from '../hooks/useAliases';
import PatternSelector from './shared/PatternSelector';

interface AnnotationDialogProps {
    isOpen: boolean;
    gridCoord: [number, number] | null;
    workAlias: string;
    weightCoordinate: string;
    onClose: () => void;
}

export default function AnnotationDialog({ 
    isOpen, 
    gridCoord, 
    weightCoordinate, 
    onClose 
}: AnnotationDialogProps) {
    const [label, setLabel] = useState('');
    const [largerPattern, setLargerPattern] = useState('');
    const [note, setNote] = useState('');
    const [isSaving, setIsSaving] = useState(false);
    const [availableLabels, setAvailableLabels] = useState<string[]>(['spurious']);
    const [availableLargerPatterns, setAvailableLargerPatterns] = useState<string[]>([]);

    const { modelAlias, inputAlias, workAlias } = useAliases();

    useEffect(() => {
        if (isOpen && gridCoord && workAlias && weightCoordinate) {
            // Load available labels for this kernel
            getKernelLabels(weightCoordinate).then(response => {
                setAvailableLabels(response.labels);
            }).catch(err => {
                console.error("Failed to load kernel labels:", err);
                setAvailableLabels(['spurious']);
            });

            // Load available larger patterns from kernel labels
            getKernelLabels(weightCoordinate).then(response => {
                setAvailableLargerPatterns(response.larger_patterns || []);
            }).catch(err => {
                console.error("Failed to load larger patterns:", err);
                setAvailableLargerPatterns([]);
            });

            // Load existing POIs for this slice
            listPOIs(modelAlias, inputAlias, workAlias, weightCoordinate).then(pois => {
                const existing = pois.find(p => p.x === gridCoord[0] && p.y === gridCoord[1]);
                if (existing) {
                    setLabel(existing.label);
                    setLargerPattern(existing.larger_pattern);
                    setNote(existing.note);
                } else {
                    setLabel('');
                    setLargerPattern('');
                    setNote('');
                }
            }).catch(err => {
                console.error("Failed to load POIs:", err);
                setLabel('');
                setLargerPattern('');
                setNote('');
            });
        }
    }, [isOpen, gridCoord, workAlias, weightCoordinate, modelAlias, inputAlias]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isOpen) {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
            return () => document.removeEventListener('keydown', handleKeyDown);
        }
    }, [isOpen, onClose]);

    const handleSave = async (e: React.MouseEvent) => {
        e.stopPropagation();
        
        if (!gridCoord || !workAlias || !weightCoordinate) {
            console.warn("Missing required data for saving POI");
            return;
        }

        setIsSaving(true);
        try {
            await savePOI(modelAlias, inputAlias, workAlias, {
                work_alias: workAlias,
                weight_coordinate: weightCoordinate,
                x: gridCoord[0],
                y: gridCoord[1],
                label,
                note,
                larger_pattern: largerPattern
            });
            onClose();
        } catch (error) {
            console.error("Save error:", error);
            alert("Failed to save POI");
        } finally {
            setIsSaving(false);
        }
    };

    const handleCancel = (e: React.MouseEvent) => {
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
                width: '320px',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
            }}>
                <h3 style={{ margin: '0 0 16px 0', color: '#fff', fontSize: '16px' }}>
                    Pixel Annotation {gridCoord ? `(${gridCoord[0]}, ${gridCoord[1]})` : ''}
                </h3>
                
                <PatternSelector
                    label="Patch Pattern"
                    options={availableLabels}
                    value={label}
                    onChange={setLabel}
                    autoFocus={true}
                    emptyMessage="No patch patterns added yet"
                    placeholder="Select a patch pattern..."
                />

                <PatternSelector
                    label="Larger Pattern"
                    options={availableLargerPatterns}
                    value={largerPattern}
                    onChange={setLargerPattern}
                    emptyMessage="No larger patterns added yet"
                    placeholder="Select a larger pattern..."
                />

                <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '4px' }}>Note</label>
                    <textarea 
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
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
                        onClick={handleCancel}
                        disabled={isSaving}
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
                        onClick={handleSave}
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
                        {isSaving ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>
        </div>
    );
}
