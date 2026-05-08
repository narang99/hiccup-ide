interface PatternTagListProps {
    title: string;
    patterns: string[];
    onRemove: (pattern: string) => void;
    isRemoving?: boolean;
    color?: string;
    emptyMessage?: string;
    preventRemoval?: (pattern: string) => boolean;
}

export default function PatternTagList({
    title,
    patterns,
    onRemove,
    isRemoving = false,
    color = '#3b82f6',
    emptyMessage = "No patterns defined",
    preventRemoval = () => false
}: PatternTagListProps) {
    return (
        <div style={{ marginBottom: '20px' }}>
            <h4 style={{ margin: '0 0 12px 0', color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
                {title}:
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {patterns.length === 0 ? (
                    <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '12px', fontStyle: 'italic' }}>
                        {emptyMessage}
                    </div>
                ) : (
                    patterns.map(pattern => (
                        <div
                            key={pattern}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                background: color,
                                color: '#fff',
                                padding: '4px 8px',
                                borderRadius: '16px',
                                fontSize: '12px',
                                fontWeight: 500,
                            }}
                        >
                            {pattern}
                            {!preventRemoval(pattern) && (
                                <button
                                    onClick={() => onRemove(pattern)}
                                    disabled={isRemoving}
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
                    ))
                )}
            </div>
        </div>
    );
}