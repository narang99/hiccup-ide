import { useState, useEffect, useCallback } from 'react';
import { getKernelNote, updateKernelNote } from '../fetchers/kernel_notes';

interface KernelNotesWidgetProps {
    weightCoordinate: string;
}

export default function KernelNotesWidget({ weightCoordinate }: KernelNotesWidgetProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [notes, setNotes] = useState('');
    const [saveTimeout, setSaveTimeout] = useState<number | null>(null);
    const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

    // Load notes when component mounts or weightCoordinate changes
    useEffect(() => {
        const loadNotes = async () => {
            try {
                const response = await getKernelNote(weightCoordinate);
                setNotes(response.notes);
            } catch (error) {
                console.error('Failed to load kernel notes:', error);
            }
        };

        loadNotes();
    }, [weightCoordinate]);

    // Debounced save function
    const debouncedSave = useCallback(async (notesToSave: string) => {
        if (saveTimeout) {
            clearTimeout(saveTimeout);
        }

        const timeout = setTimeout(async () => {
            setSaveStatus('saving');
            try {
                await updateKernelNote(weightCoordinate, notesToSave);
                setSaveStatus('saved');
                setTimeout(() => setSaveStatus('idle'), 2000);
            } catch (error) {
                console.error('Failed to save kernel notes:', error);
                setSaveStatus('error');
                setTimeout(() => setSaveStatus('idle'), 3000);
            }
        }, 2000);

        setSaveTimeout(timeout);
    }, [weightCoordinate, saveTimeout]);

    // Handle notes change with debounced save
    const handleNotesChange = (newNotes: string) => {
        setNotes(newNotes);
        if (newNotes !== notes) {
            debouncedSave(newNotes);
        }
    };

    // Get save status indicator
    const getSaveStatusIndicator = () => {
        switch (saveStatus) {
            case 'saving':
                return { text: 'Saving...', color: 'rgba(59, 130, 246, 0.8)' };
            case 'saved':
                return { text: 'Saved ✓', color: 'rgba(34, 197, 94, 0.8)' };
            case 'error':
                return { text: 'Save failed ✗', color: 'rgba(239, 68, 68, 0.8)' };
            default:
                return null;
        }
    };

    const statusIndicator = getSaveStatusIndicator();

    if (isOpen) {
        return (
            <div style={{
                position: 'fixed',
                bottom: '20px',
                right: '20px',
                width: '320px',
                background: 'rgba(13, 13, 20, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                zIndex: 1000,
            }}>
                {/* Header */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 16px',
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                    }}>
                        <span style={{
                            fontSize: '14px',
                            fontWeight: 600,
                            color: '#fff',
                        }}>
                            📝 Kernel Notes
                        </span>
                        {statusIndicator && (
                            <span style={{
                                fontSize: '10px',
                                color: statusIndicator.color,
                                fontWeight: 500,
                            }}>
                                {statusIndicator.text}
                            </span>
                        )}
                    </div>
                    <button
                        onClick={() => setIsOpen(false)}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: 'rgba(255,255,255,0.6)',
                            cursor: 'pointer',
                            fontSize: '16px',
                            padding: '4px',
                            borderRadius: '4px',
                        }}
                    >
                        ✕
                    </button>
                </div>

                {/* Content */}
                <div style={{ padding: '16px' }}>
                    <textarea
                        value={notes}
                        onChange={(e) => handleNotesChange(e.target.value)}
                        placeholder="Add notes about this kernel..."
                        style={{
                            width: '100%',
                            height: '160px',
                            background: '#0d0d14',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                            padding: '12px',
                            color: '#fff',
                            fontSize: '13px',
                            fontFamily: 'system-ui, -apple-system, sans-serif',
                            resize: 'vertical',
                            outline: 'none',
                            minHeight: '80px',
                            maxHeight: '300px',
                            boxSizing: 'border-box',
                        }}
                    />
                    <div style={{
                        marginTop: '8px',
                        fontSize: '11px',
                        color: 'rgba(255,255,255,0.4)',
                    }}>
                        Notes auto-save after 2 seconds
                    </div>
                </div>
            </div>
        );
    }

    return (
        <button
            onClick={() => setIsOpen(true)}
            style={{
                position: 'fixed',
                bottom: '20px',
                right: '20px',
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                background: 'rgba(13, 13, 20, 0.88)',
                border: '1px solid rgba(255,255,255,0.09)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
                color: 'rgba(255,255,255,0.75)',
                fontSize: '18px',
                cursor: 'pointer',
                zIndex: 1000,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scale(1.05)';
                e.currentTarget.style.boxShadow = '0 6px 25px rgba(0,0,0,0.5)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
                e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.4)';
            }}
        >
            📝
        </button>
    );
}